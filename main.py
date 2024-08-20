import torch
from transformer import *

device = torch.device('mps')
torch.manual_seed(1337)

model = Transformer(embd_sze=384, src_vocab_sze=64, trgt_vocab_sze=64, max_seq_len=256)
model.to(device=device)
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))