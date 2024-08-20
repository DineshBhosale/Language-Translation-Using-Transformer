import re
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformer import *
from unicodedata import normalize
from torch.nn import functional as F

device = torch.device('mps')
torch.manual_seed(1337)

model = Transformer(embd_sze=384, src_vocab_sze=64, trgt_vocab_sze=64, max_seq_len=256)
model.to(device=device)
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))