import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np

# -------------------------------
# Hyperparameters and Device
# -------------------------------
HIDDEN_SIZE = 1024    # hidden dimension for each network
EMBED_SIZE = 64       # embedding size (character representation)
NUM_BITS = 10         # for binary addressing memory (2^10 memory slots)
SEQ_LENGTH = 100      # sequence length for training
BATCH_SIZE = 48       # batch size
NUM_EPOCHS = 30       # number of training epochs
LR = 0.003            # learning rate
SAMPLE_EVERY = 50     # sample text every N batches
BATCHES_PER_EPOCH = 200
NUM_SUB = 3           # Number of subnetworks
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Data Preparation (Shakespeare)
# -------------------------------
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - SEQ_LENGTH, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LENGTH] for i in ix])
    y = torch.stack([data[i+1:i+SEQ_LENGTH+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# -------------------------------
# Binary Memory RNN Cell (Base)
# -------------------------------
class BinaryMemoryRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_bits=NUM_BITS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bits = num_bits
        self.mem_size = 2 ** num_bits
        
        # Memory components
        self.memory_buffer = deque(maxlen=self.mem_size)
        self.register_buffer('binary_powers', 2 ** torch.arange(num_bits-1, -1, -1).float())
        
        # Network parameters
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.M = nn.Linear(hidden_size, num_bits * 2)
        self.Q_recent = nn.Linear(hidden_size, hidden_size)
        self.Q_long = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def init_memory(self, batch_size):
        self.memory_buffer.clear()
        # Initialize with zero state
        self.memory_buffer.append(torch.zeros(batch_size, self.hidden_size, device=DEVICE))

    def reset_memory(self):
        self.memory_buffer.clear()

    def forward(self, x, h_prev, tau=1.0):
        # Address generation
        logits = self.M(h_prev)
        logits1, logits2 = logits.chunk(2, dim=-1)

        # Gumbel-Softmax sampling
        if self.training:
            def gumbel_binary(logits_part):
                g = -torch.log(-torch.log(torch.rand_like(logits_part)))
                return torch.sigmoid((logits_part + g) / tau)
            binary1 = (gumbel_binary(logits1) > 0.5).float()
            binary2 = (gumbel_binary(logits2) > 0.5).float()
        else:
            binary1 = (torch.sigmoid(logits1) > 0.5).float()
            binary2 = (torch.sigmoid(logits2) > 0.5).float()

        # Memory indexing
        index1 = (binary1 * self.binary_powers).sum(dim=1).long()
        index2 = (binary2 * self.binary_powers).sum(dim=1).long()
        
        # Memory retrieval
        if len(self.memory_buffer) > 0:
            # Convert deque to tensor: (T, B, H)
            mem_tensor = torch.stack(list(self.memory_buffer))
            T = mem_tensor.size(0)
            
            # Clamp indices and gather
            index1 = torch.clamp(index1, 0, T-1)
            index2 = torch.clamp(index2, 0, T-1)
            
            # Advanced indexing: [index1, batch_indices, :]
            batch_indices = torch.arange(x.size(0), device=DEVICE)
            h_mem_recent = mem_tensor[index1, batch_indices]
            h_mem_long = mem_tensor[index2, batch_indices]
        else:
            h_mem_recent = torch.zeros_like(h_prev)
            h_mem_long = torch.zeros_like(h_prev)
        # State update
        pre_act = self.W(x) + self.U(h_prev) + self.Q_recent(h_mem_recent) + self.Q_long(h_mem_long)
        h_new = torch.sigmoid(self.ln(pre_act))
        
        # Update memory
        self.memory_buffer.append(h_new.detach().clone())
        return h_new

# -------------------------------
# Master RNN Cell with Gating
# -------------------------------
class MasterBinaryMemoryRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_sub=NUM_SUB):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sub = num_sub
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_size, num_sub)
        self.fusion = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, prev_master_h, prev_sub_h_list, sub_cells, tau=1.0):
        # 1. Compute gating weights using previous master state
        gate_logits = self.gate(prev_master_h)
        gate_onehot = F.gumbel_softmax(gate_logits, tau=tau, hard=True)
        
        # 2. Process input through all subnetworks
        new_sub_h_list = []
        for cell, h_prev in zip(sub_cells, prev_sub_h_list):
            new_sub_h = cell(x, h_prev)
            new_sub_h_list.append(new_sub_h)
        
        # 3. Fuse subnetwork states using gating weights
        sub_states = torch.stack(new_sub_h_list)  # [num_sub, B, H]
        weighted_states = torch.einsum('bs,sbh->bh', gate_onehot, sub_states)
        
        # 4. Update master state
        fused = self.fusion(weighted_states)
        master_h_new = torch.sigmoid(self.ln(prev_master_h + fused))
        
        return master_h_new, gate_onehot, new_sub_h_list

# -------------------------------
# Multi-Network Char RNN Model
# -------------------------------
class MultiNetworkCharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_sub=NUM_SUB):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.num_sub = num_sub
        
        # Initialize cells
        self.sub_cells = nn.ModuleList([
            BinaryMemoryRNNCell(embed_size, hidden_size) 
            for _ in range(num_sub)
        ])
        self.master_cell = MasterBinaryMemoryRNNCell(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_memory(self, batch_size):
        for cell in self.sub_cells:
            cell.init_memory(batch_size)

    def reset_memory(self):
        for cell in self.sub_cells:
            cell.reset_memory()

    def forward(self, x, master_h, sub_h_list, tau=1.0):
        x_emb = self.embed(x)
        master_h_new, gate_onehot, new_sub_h_list = self.master_cell(
            x_emb, master_h, sub_h_list, self.sub_cells, tau=tau
        )
        logits = self.fc(master_h_new)
        return logits, master_h_new, new_sub_h_list, gate_onehot

# -------------------------------
# Training Utilities
# -------------------------------
def generate_sample(model, start_char='\n', length=200, temp=0.8):
    model.eval()
    model.reset_memory()
    with torch.no_grad():
        master_h = torch.zeros(1, HIDDEN_SIZE, device=DEVICE)
        sub_h_list = [torch.zeros(1, HIDDEN_SIZE, device=DEVICE) for _ in range(NUM_SUB)]
        input_idx = torch.tensor([char_to_idx[start_char]], device=DEVICE)
        generated = [start_char]
        
        for _ in range(length):
            logits, master_h, sub_h_list, _ = model(input_idx, master_h, sub_h_list, temp)
            probs = F.softmax(logits.squeeze()/temp, dim=-1)
            input_idx = torch.multinomial(probs, 1)
            generated.append(idx_to_char[input_idx.item()])
        
        print(''.join(generated))
        print("\n" + "-"*50 + "\n")
    model.train()

# -------------------------------
# Training Loop
# -------------------------------
model = MultiNetworkCharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Initialize memories
model.init_memory(BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    model.train()
    
    for batch_idx in range(BATCHES_PER_EPOCH):
        x_batch, y_batch = get_batch()
        model.reset_memory()
        
        master_h = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, device=DEVICE)
        sub_h_list = [torch.zeros(BATCH_SIZE, HIDDEN_SIZE, device=DEVICE) 
                     for _ in range(NUM_SUB)]
        
        tau = max(0.5, 1.0 - epoch*0.03)  # Anneal temperature
        batch_loss = 0
        
        optimizer.zero_grad()
        for t in range(SEQ_LENGTH):
            logits, master_h, sub_h_list, _ = model(x_batch[:, t], master_h, sub_h_list, tau)
            loss = criterion(logits, y_batch[:, t])
            batch_loss += loss
        
        (batch_loss/SEQ_LENGTH).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        avg_loss = batch_loss.item()/SEQ_LENGTH
        total_loss += avg_loss
        
        if batch_idx % SAMPLE_EVERY == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx} | Loss: {avg_loss:.4f}")
            generate_sample(model)
    
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/BATCHES_PER_EPOCH:.4f}")
