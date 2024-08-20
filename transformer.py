import torch
import torch.nn as nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, block_sze, embd_sze):
        super().__init__()
        
        self.block_sze = block_sze
        self.embd_sze = embd_sze
        self.register_buffer('pe', self.init_pe().unsqueeze(0))

    def init_pe(self):

        p_e = torch.zeros(self.block_sze, self.embd_sze)
        positions = torch.arange(self.block_sze, dtype=torch.float).unsqueeze(1)
        div = torch.pow(10000, 2 * torch.arange(0, self.embd_sze, 2).float() / self.embd_sze)

        p_e[:, 0::2] = torch.sin(positions * div)
        p_e[:, 1::2] = torch.cos(positions * div)

        return p_e

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Head(nn.Module):
    # single masked self-attention head
    def __init__(self, block_sze, embd_sze, head_size, dropout=0.2, attention_mask = False):
        super().__init__()
        self.key = nn.Linear(embd_sze, head_size)
        self.query = nn.Linear(embd_sze, head_size)
        self.value = nn.Linear(embd_sze, head_size)
        self.attention_mask = attention_mask
        if self.attention_mask:
            self.register_buffer('tril', torch.tril(torch.ones(block_sze, block_sze)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):

        B, T, C = Q.shape

        k = self.key(K) # (B, T, C) -> (B, T, hs)
        q = self.query(Q)   # (B, T, C) -> (B, T, hs)
        v = self.value(V)   # (B, T, C) -> (B, T, hs)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1])**(-0.5)    # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.attention_mask:
            wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v   # (B, T, T) @ (B, T, hs) -> (B, T, hs) 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, block_sze, embd_sze, num_heads, head_size, dropout=0.2, attention_mask=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_sze, embd_sze, head_size, dropout=dropout, attention_mask=attention_mask) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_sze)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x, x, x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class MultiCrossHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, block_sze, embd_sze, num_heads, head_size, dropout=0.2, attention_mask=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_sze, embd_sze, head_size, dropout=dropout, attention_mask=attention_mask) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_sze)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output):
        out = torch.cat([h(x, encoder_output, encoder_output) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    "feed forward network"

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-5):
        super().__init__()

        shape = (1, num_features)
        self.eps = eps

        # scale and shift parameters (trained with backprop)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):
        mean = X.mean(axis = 1, keepdim = True)
        var = X.var(axis = 1, keepdim = True)

        X_hat = (X - mean) / (torch.sqrt(var) + self.eps)

        return self.gamma * X_hat + self.beta

class Encoder(nn.Module):
    def __init__(self, block_sze, embd_sze, num_heads):
        # embd_sze: number of embeddings in embedding dimension, num_heads: the number of heads 
        super().__init__()

        head_size = embd_sze // num_heads
        self.sa = MultiHeadAttention(block_sze, embd_sze, num_heads, head_size, attention_mask=False)
        self.ln1 = LayerNorm(embd_sze)
        self.ffwd = FeedFoward(embd_sze)
        self.ln2 = LayerNorm(embd_sze)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, block_sze, embd_size, num_heads):
        super().__init__()

        head_size = embd_size // num_heads
        self.sa = MultiHeadAttention(block_sze, embd_size, num_heads, head_size, attention_mask=True)
        self.ln1 = LayerNorm(embd_size)
        
        self.ca = MultiCrossHeadAttention(block_sze, embd_size, num_heads, head_size, attention_mask=False)
        self.ln2 = LayerNorm(embd_size)

        self.ffwd = FeedFoward(embd_size)
        self.ln3 = LayerNorm(embd_size)

    def forward(self, x, encoder_output):
        x = x + self.ln1(self.sa(x))
        x = x + self.ln2(self.ca(x, encoder_output, encoder_output))
        x = x + self.ln3(self.ffwd(x))
        return x

class Transformer(nn.Module):
    def __init__(self, embd_sze, src_vocab_sze, trgt_vocab_sze, max_seq_len):
        super().__init__()

        self.n_layers = 6
        self.num_heads = 6
        self.position_embedding = PositionalEncoding(max_seq_len, embd_sze)
        
        # encoder
        self.enc_token_embedding = nn.Embedding(src_vocab_sze, embd_sze)

        self.encoder = nn.Sequential(*[Encoder(src_vocab_sze, embd_sze, self.num_heads) for _ in range(self.n_layers)])
        
        # decoder
        self.dec_token_embedding = nn.Embedding(trgt_vocab_sze, embd_sze)
        self.decoder = nn.Sequential(*[Decoder(trgt_vocab_sze, embd_sze, self.num_heads) for _ in range(self.n_layers)])

        self.lm_head = nn.Linear(embd_sze, trgt_vocab_sze)
    
    def forward(self, src, trgt):
        src = self.enc_token_embedding(src) + self.position_embedding(src)
        encoder_output = self.encoder(src)

        trgt = self.dec_token_embedding(trgt) + self.position_embedding(trgt)
        trgt = self.decoder(trgt, encoder_output)

        out = self.lm_head(trgt)
        return out 