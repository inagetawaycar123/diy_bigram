import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):

        logits = self.token_embedding_table(idx) # (B,T,C)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx):

        logits, loss = self(idx)

        logits = logits[:, -1, :] # becomes (B, C)
        probs = F.softmax(logits, dim=-1) # (B, C)