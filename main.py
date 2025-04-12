import torch
import torch.nn as nn
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Hyperparams 
HIDDEN_SIZE = 512  
EMBED_SIZE = 64     
NUM_BITS = 10
SEQ_LENGTH = 100    
BATCH_SIZE = 48 
NUM_EPOCHS = 30
LR = 0.003 
SAMPLE_EVERY = 50   
BATCHES_PER_EPOCH = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BinaryMemoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_bits=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bits = num_bits
        self.mem_size = 2 ** num_bits
        
        # Single weight matrix for both addresses
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.M = nn.Linear(hidden_size, num_bits * 2)  # Output 2*num_bits
        self.Q_recent = nn.Linear(hidden_size, hidden_size)
        self.Q_long = nn.Linear(hidden_size, hidden_size)
        
        self.ln = nn.LayerNorm(hidden_size)
        self.memory_buffer = deque(maxlen=self.mem_size)
        self.register_buffer('binary_powers', 2**torch.arange(num_bits-1, -1, step=-1))

    def forward(self, x, h_prev, tau=1.0):
        # Single matrix produces both addresses
        logits = self.M(h_prev)
        logits1, logits2 = logits.chunk(2, dim=-1)  # Split into two addresses

        if self.training:
            def gumbel_binary(logits_part):
                noise = torch.rand_like(logits_part).log_().neg_().log_().neg_()
                binary_soft = torch.sigmoid((logits_part + noise) / tau)
                binary_hard = (binary_soft > 0.5).float()
                return binary_hard - binary_soft.detach() + binary_soft
            
            binary1 = gumbel_binary(logits1)
            binary2 = gumbel_binary(logits2)
        else:
            binary1 = (torch.sigmoid(logits1) > 0.5).float()
            binary2 = (torch.sigmoid(logits2) > 0.5).float()

        # Index calculation with constraints
        index1 = (binary1 * self.binary_powers).sum(dim=1).long()
        index2 = (binary2 * self.binary_powers).sum(dim=1).long()
        
        if len(self.memory_buffer) > 1:
            split_idx = len(self.memory_buffer) // 2
            index1 = torch.clamp(index1, 0, split_idx-1)
            index2 = torch.clamp(index2, split_idx, len(self.memory_buffer)-1)
            
            h_mem_recent = torch.stack([self.memory_buffer[i] for i in index1.tolist()])
            h_mem_long = torch.stack([self.memory_buffer[i] for i in index2.tolist()])
        else:
            h_mem_recent = torch.zeros_like(h_prev)
            h_mem_long = torch.zeros_like(h_prev)

        h_new = torch.sigmoid(self.ln(
            self.W(x) + 
            self.U(h_prev) + 
            self.Q_recent(h_mem_recent) + 
            self.Q_long(h_mem_long)
        ))

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
loss_history = []

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
        loss_history.append(avg_batch_loss)
        if batch_idx % 2 == 0:
            print(f"Batch {batch_idx+1} | Loss: {avg_batch_loss}")
        if (batch_idx + 1) % SAMPLE_EVERY == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {avg_batch_loss:.4f}")
            generate_sample(model)
    
    print(f"Epoch {epoch+1} Complete | Avg Loss: {total_loss/BATCHES_PER_EPOCH:.4f}")
plt.figure(figsize=(10, 5))
plt.plot(loss_history, alpha=0.7)
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
