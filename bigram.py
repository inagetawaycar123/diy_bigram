import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
