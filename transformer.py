import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, block_sze, embd_sze):
        super(PositionalEncoding, self).__init__()
        
        self.block_sze = block_sze
        self.embd_sze = embd_sze
        p_e = torch.zeros(self.block_sze, self.embd_sze)
        positions = torch.arange(self.block_sze, dtype=torch.float).unsqueeze(1)
        den = torch.exp(- torch.arange(0, self.embd_sze, 2) * math.log(10000) / self.embd_sze)

        p_e[:, 0::2] = torch.sin(positions * den)
        p_e[:, 1::2] = torch.cos(positions * den)
        p_e = p_e.unsqueeze(0)

        self.register_buffer('p_e', p_e)

    def forward(self, x):
        return x + self.p_e[:, :x.size(1)]

class Head(nn.Module):
    # single masked self-attention head
    def __init__(self, block_sze, embd_sze, head_size, dropout=0.2):
        super(Head, self).__init__()
        self.key = nn.Linear(embd_sze, head_size)
        self.query = nn.Linear(embd_sze, head_size)
        self.value = nn.Linear(embd_sze, head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None):

        #B, T, C = Q.shape

        k = self.key(K) # (B, T, C) -> (B, T, hs)
        q = self.query(Q)   # (B, T, C) -> (B, T, hs)
        v = self.value(V)   # (B, T, C) -> (B, T, hs)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1])**(-0.5)    # (B, T, C) @ (B, C, T) -> (B, T, T)
        if mask is not None:
            wei = wei.masked_fill(mask==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v   # (B, T, T) @ (B, T, hs) -> (B, T, hs) 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, block_sze, embd_sze, num_heads, head_size, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(block_sze, embd_sze, head_size, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_sze)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask):
        out = torch.cat([h(x, x, x, src_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class MultiCrossHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, block_sze, embd_sze, num_heads, head_size, dropout=0.2):
        super(MultiCrossHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(block_sze, embd_sze, head_size, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_sze)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask):
        out = torch.cat([h(x, encoder_output, encoder_output, src_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    "feed forward network"

    def __init__(self, n_embd, dropout=0.2):
        super(FeedFoward, self).__init__()
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
        super(LayerNorm, self).__init__()

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
        super(Encoder, self).__init__()

        head_size = embd_sze // num_heads
        self.sa = MultiHeadAttention(block_sze, embd_sze, num_heads, head_size)
        self.ln1 = LayerNorm(embd_sze)
        self.ffwd = FeedFoward(embd_sze)
        self.ln2 = LayerNorm(embd_sze)

    def forward(self, x, src_mask):
        x = x + self.sa(self.ln1(x), src_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, block_sze, embd_sze, num_heads):
        super(Decoder, self).__init__()

        head_size = embd_sze // num_heads
        self.sa = MultiHeadAttention(block_sze, embd_sze, num_heads, head_size)
        self.ln1 = LayerNorm(embd_sze)
        
        self.ca = MultiCrossHeadAttention(block_sze, embd_sze, num_heads, head_size)
        self.ln2 = LayerNorm(embd_sze)

        self.ffwd = FeedFoward(embd_sze)
        self.ln3 = LayerNorm(embd_sze)

    def forward(self, x, e_output, src_mask, tgt_mask):
        x = x + self.ln1(self.sa(x, tgt_mask))
        x = x + self.ln2(self.ca(x, e_output, src_mask))
        x = x + self.ln3(self.ffwd(x))
        return x

class Transformer(nn.Module):
    def __init__(self, embd_sze, src_vocab_sze, tgt_vocab_sze, max_seq_len):
        super(Transformer, self).__init__()

        self.n_layers = 6
        self.num_heads = 6
        self.position_embedding = PositionalEncoding(max_seq_len, embd_sze)
        
        # encoder
        self.enc_token_embedding = nn.Embedding(src_vocab_sze, embd_sze)

        self.encoder = nn.ModuleList([Encoder(src_vocab_sze, embd_sze, self.num_heads) for _ in range(self.n_layers)])
        
        # decoder
        self.dec_token_embedding = nn.Embedding(tgt_vocab_sze, embd_sze)
        self.decoder = nn.ModuleList([Decoder(tgt_vocab_sze, embd_sze, self.num_heads) for _ in range(self.n_layers)])

        self.lm_head = nn.Linear(embd_sze, tgt_vocab_sze)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        e_output = self.position_embedding(self.enc_token_embedding(src))
        for encoder in self.encoder:
            e_output = encoder(e_output, src_mask)

        d_output = self.position_embedding(self.dec_token_embedding(tgt))
        for decoder in self.decoder:
            d_output = decoder(d_output, e_output, src_mask, tgt_mask)

        out = self.lm_head(d_output)
        return out 