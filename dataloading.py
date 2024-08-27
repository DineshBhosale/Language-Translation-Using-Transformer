import re
import torch
import torch.utils.data as data
from unicodedata import normalize

class Vocabulary():
    def __init__(self, type):
        self.type = type
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unkn_id = 3
        self.stoi = {} 
        self.itos = {}

    def generate_vocabulary(self, l_sentences):
        counter = {}
        for sentence in l_sentences:
            for word in sentence:
                counter[word] = 1 + counter.get(word, 0)

        counter = sorted(counter, key = counter.values, reverse=True)
        
        for word in counter.keys():
            self.stoi[word] = len(self.stoi) + 1
            self.itos[len(self.itos) + 1] = word
                
    def encode_str(self, sentence):
        tokens = []
        for word in sentence.split():
            if word not in self.stoi:
                token = self.unkn_id
            elif word == '[start]':
                token = self.bos_id
            elif word == '[end]':
                token = self.eos_id
            else:    
                token = self.stoi[word]         

            tokens.append(token)
        return tokens

    def decode_tokens(self, tokens):
        ret = ""
        for token in tokens:
            if token in self.itos:
                ret += self.itos[token] + " "
            else:
                ret += "unknown " 
        return ret

    def get_mask(self, sequence, device=None):
        mask = (sequence != self.pad_id).unsqueeze(1)

        if self.type == "tgt":
            attn_mask = torch.tril(torch.ones((1, sequence.shape[1], sequence.shape[1]), dtype=torch.bool)).to(device)
            mask = mask & attn_mask
            
        return mask

    def save_stoi_vocab(self):
        f = open("./{0}-stoi-vocab.txt".format(self.type), "w")
        for word in self.stoi:
            line = word + ":" + str(self.stoi[word])
            f.write(line)
        f.close()

    def save_itos_vocab(self):
        f = open("./{0}-itos-vocab.txt".format(self.type), "w")
        for token in self.itos:
            line = token + ":" + str(self.itos[token])
            f.write(line)
        f.close()
        
    def read_stoi_vocab(self, path):
        f = open(path, "r")
        while True:
            line = f.readline()
            if not line:
                break
                
            vals = line.split(":")
            self.stoi[vals[0]] = int(vals[1])
        f.close()

    def read_itos_vocab(self, path):
        f = open(path, "r")
        while True:
            line = f.readline()
            if not line:
                break
                
            vals = line.split(":")
            self.itos[vals[0]] = int(vals[1])
        f.close()

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
    text = '[start] ' + clean_text(text) + ' [end]'
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