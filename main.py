import torch
import torch.nn as nn
from collections import deque
import numpy as np

# Hyperparameters ==============================================================
HIDDEN_SIZE = 1024  # Reduced from 2048: Better stability vs capacity tradeoff
EMBED_SIZE = 64     # From 128: Embedding dimension (character representation size)
NUM_BITS = 10       # Increased from 8: 1024 memory slots (2^10)
SEQ_LENGTH = 100    # Reduced from 150: Better for gradient flow
BATCH_SIZE = 48     # Slightly reduced for memory safety
NUM_EPOCHS = 30     # Increased to allow longer training
LR = 0.003          # Reduced from 0.007: Smoother convergence
SAMPLE_EVERY = 50   
BATCHES_PER_EPOCH = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Embedding Dimension Explained -------------------------------------------------
# EMBED_SIZE = 64 means each character is represented by a 64-dimensional vector
# Think of it like a "personality vector" for each character that the network learns
# Higher values = more nuanced representations but more parameters to learn

# 1. Enhanced Binary Memory RNN -------------------------------------------------
class BinaryMemoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_bits=10):  # Note num_bits change
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bits = num_bits
        self.mem_size = 2 ** num_bits  # Now 1024 slots
        
        # Learnable transformations
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.M = nn.Linear(hidden_size, num_bits)
        
        # New: LayerNorm for stability
        self.ln = nn.LayerNorm(hidden_size)
        
        # Memory storage
        self.memory_buffer = deque(maxlen=self.mem_size)
        self.register_buffer('binary_powers', 2**torch.arange(num_bits-1, -1, step=-1))
        
    def forward(self, x, h_prev, tau=1.0):
        # Address generation remains same
        logits = self.M(h_prev)
        
        if self.training:
            noise = torch.rand_like(logits).log_().neg_().log_().neg_()
            binary_soft = torch.sigmoid((logits + noise) / tau)
            binary_hard = (binary_soft > 0.5).float()
            binary = binary_hard - binary_soft.detach() + binary_soft
        else:
            binary = (torch.sigmoid(logits) > 0.5).float()
        
        index = (binary * self.binary_powers).sum(dim=1).long()
        
        # Memory retrieval with enhanced safety
        if len(self.memory_buffer) == 0:
            h_mem = torch.zeros_like(h_prev)
        else:
            index = torch.clamp(index, 0, len(self.memory_buffer)-1)
            # Convert indices to CPU for deque access
            index_cpu = index.cpu()
            valid_indices = index_cpu.tolist() if index.dim() else [index_cpu.item()]
            
            h_mem = torch.stack([self.memory_buffer[i] for i in valid_indices]).to(x.device)
        
        # LayerNorm added here
        h_new = torch.sigmoid(self.ln(
            self.W(x) + self.U(h_prev) + self.Q(h_mem)
        ))
        
        # Update memory
        self.memory_buffer.extend(h_new.detach().unbind())
        
        return h_new
    
    def reset_memory(self):
        self.memory_buffer.clear()

# 2. Character Model -----------------------------------------------------------
class CharMemoryRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE)
        self.rnn = BinaryMemoryRNN(EMBED_SIZE, HIDDEN_SIZE, NUM_BITS)
        self.fc = nn.Linear(HIDDEN_SIZE, vocab_size)
        
    def forward(self, x, h_prev, tau=1.0):
        x_emb = self.embed(x)
        h_new = self.rnn(x_emb, h_prev, tau)
        return self.fc(h_new), h_new
    
    def reset_memory(self):
        self.rnn.reset_memory()

# Remaining code identical to previous version except batch size changes =========

# 3. Data Preparation ---------------------------------------------------------
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - SEQ_LENGTH, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LENGTH] for i in ix])
    y = torch.stack([data[i+1:i+SEQ_LENGTH+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# 4. Training Utilities ------------------------------------------------------
def generate_sample(model, start_char='\n', length=200, temp=0.8):
    model.eval()
    model.reset_memory()
    with torch.no_grad():
        h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
        input_idx = torch.tensor([char_to_idx[start_char]]).to(DEVICE)  # Fix: remove extra dimension
        generated = [start_char]
        
        for _ in range(length):
            logits, h = model(input_idx, h)
            probs = torch.softmax(logits.squeeze() / temp, dim=-1)
            input_idx = torch.multinomial(probs, num_samples=1)
            generated.append(idx_to_char[input_idx.item()])
            
    print(''.join(generated))
    print("\n" + "-"*50 + "\n")
    model.train()

# 5. Training Loop -----------------------------------------------------------
model = CharMemoryRNN(vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    model.train()
    
    for batch_idx in range(BATCHES_PER_EPOCH):
        x, y = get_batch()
        h = torch.zeros(BATCH_SIZE, HIDDEN_SIZE).to(DEVICE)
        model.reset_memory()
        tau = max(1.0 - (epoch*BATCHES_PER_EPOCH + batch_idx)/1000, 0.1)
        batch_loss = 0
        
        for t in range(SEQ_LENGTH):
            logits, h = model(x[:, t], h, tau)
            h = h.detach()
            
            loss = criterion(logits.view(-1, vocab_size), y[:, t].view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            batch_loss += loss.item()
        
        avg_batch_loss = batch_loss / SEQ_LENGTH
        total_loss += avg_batch_loss
        if batch_idx % 2 == 0:
            print(f"Batch {batch_idx+1} | Loss: {avg_batch_loss}")
        if (batch_idx + 1) % SAMPLE_EVERY == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {avg_batch_loss:.4f}")
            generate_sample(model)
    
    print(f"Epoch {epoch+1} Complete | Avg Loss: {total_loss/BATCHES_PER_EPOCH:.4f}")
