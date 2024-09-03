import torch
import pandas as pd
from utils.transformer import *
from utils.dataloading import *
from tqdm import tqdm
from timeit import default_timer as timer

device = torch.device('mps')
torch.manual_seed(1337)
torch.mps.set_per_process_memory_fraction(0.0)

max_seq_length = 28
vocab_s = Vocabulary("src")
vocab_s.read_vocab()

vocab_t = Vocabulary("tgt")
vocab_t.read_vocab()

src_vocab_size, tgt_vocab_size = len(vocab_s.stoi), len(vocab_t.stoi)
print("Source Vocab Size is {0} and Target Vocab Size is {1}".format(src_vocab_size, tgt_vocab_size))

model = Transformer(embd_sze=512, src_vocab_sze=src_vocab_size, tgt_vocab_sze=tgt_vocab_size, max_seq_len=max_seq_length)
model.to(device)
model.eval()
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))

model.load_state_dict(torch.load("./weights/model-iteration-02.pt", weights_only=True))

def inference(e_output, src_mask):
    tgt_input = torch.LongTensor([vocab_t.pad_id] * max_seq_length).to(device)
    tgt_input[0] = vocab_t.bos_id
    tgt_input =  tgt_input.reshape(1, -1)
    tgt_mask = vocab_t.get_mask(tgt_input, device)

    tgt_output = []

    for i in range(max_seq_length-1):
        output = model.decode(tgt_input, e_output, src_mask, tgt_mask)
        output = torch.argmax(output, dim=-1)

        last_word_id = output[0][i].item()

        if last_word_id == vocab_t.eos_id:
            break

        if last_word_id == vocab_t.pad_id:
            continue

        tgt_input[0][i+1] = last_word_id
        tgt_output.append(last_word_id)

    return tgt_output
    
src = "the weather is rainy"
src = torch.LongTensor(vocab_s.encode_str(src)).to(device)
src = src.reshape(1, -1)
src_mask = vocab_s.get_mask(src)
e_output = model.encode(src, src_mask)
d_output = inference(e_output, src_mask)

print(vocab_s.decode_tokens(src.tolist()[0]) + '\t' + vocab_t.decode_tokens(d_output))