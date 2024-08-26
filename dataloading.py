import re
import torch
import torch.utils.data as data
from unicodedata import normalize

class TranslationDataset(data.Dataset):
    def __init__(self, src_tokens, tgt_tokens, max_len):
        super().__init__()
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.max_len = max_len

    def __len__(self):
        return len(self.src_tokens)
    
    def __getitem__(self, index):
        src_data = self.src_tokens[index] + [0] * (self.max_len - len(self.src_tokens[index])) # post zero padding
        tgt_data = self.tgt_tokens[index] + [0] * (self.max_len - len(self.tgt_tokens[index])) # post zero padding
        return torch.LongTensor(src_data), torch.LongTensor(tgt_data)

def create_vocab(sentence, stoi_vocab, itos_vocab):
    tokens = sentence.split()
    for token in tokens:
        if token not in stoi_vocab:
            enc = len(stoi_vocab)+1
            stoi_vocab[token]=enc
            itos_vocab[enc] = token

def tokenize_sentence(sentence, stoi_vocab):
    tokens = sentence.split()
    token_ids = []
    for token in tokens:
        token_ids.append(stoi_vocab[token])
    return token_ids

def clean_text(text):
    text = normalize('NFD', text.lower())
    text = re.sub('[^A-Za-z ]+', ' ', text)
    return text

def clean_prepare_text(text):
    text = '[start]' + clean_text(text) + '[end]'
    return text

def make_mask(src, tgt, src_vocab, tgt_vocab, device):
    src_mask = (src != src_vocab["<pad>"]).unsqueeze(1)
    tgt_mask = (tgt != tgt_vocab["<pad>"]).unsqueeze(1)
    
    dec_attention_mask = torch.tril(torch.ones((1, tgt.shape[1], tgt.shape[1]), dtype=torch.bool)).to(device)
    tgt_mask = tgt_mask & dec_attention_mask
    return src_mask, tgt_mask

def print_sample_data(df, sample):
    for i, (src_sent, tgt_sent) in enumerate(zip(df['src'], df['tgt'])):
        if i == sample:
            break 
        print(src_sent + "\t\t\t" + tgt_sent)