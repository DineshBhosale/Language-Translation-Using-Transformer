import re
import torch
import pickle
import torch.utils.data as data
from unicodedata import normalize

class Vocabulary():
    def __init__(self, type):
        self.type = type
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unkn_id = 3
        self.stoi = {'<pad>': self.pad_id, '[start]': self.bos_id, '[end]': self.eos_id} 
        self.itos = {self.pad_id: '<pad>', self.bos_id: '[start]', self.eos_id: '[end]'}

    def generate_vocabulary(self, l_sentences):
        counter = {}
        for sentence in l_sentences:
            words = sentence.split()
            for word in words:
                if word == '[start]':
                    continue
                elif word == '[end]':
                    continue
                else:
                    counter[word] = 1 + counter.get(word, 0)

        counter = dict(sorted(counter.items(), key = lambda x: x[1], reverse=True))
        
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
            elif word == '<pad>':
                token = self.pad_id
            else:    
                token = self.stoi[word]         

            tokens.append(token)
        return tokens

    def decode_tokens(self, tokens):
        ret = ""
        for token in tokens:
            if token in self.itos:
                if token is self.bos_id\
                    or token is self.pad_id\
                        or token is self.eos_id:
                    continue
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

    def save_vocab(self):

        with open('./{0}-stoi-vocab.pkl'.format(self.type), 'wb') as f:
            pickle.dump(self.stoi, f)

        with open('./{0}-itos-vocab.pkl'.format(self.type), 'wb') as f:
            pickle.dump(self.itos, f)
        
    def read_vocab(self):
        with open('./{0}-stoi-vocab.pkl'.format(self.type), 'rb') as f:
            self.stoi = pickle.load(f)

        with open('./{0}-itos-vocab.pkl'.format(self.type), 'rb') as f:
            self.itos = pickle.load(f)

class TranslationDataset(data.Dataset):
    def __init__(self, src_tokens, tgt_tokens, max_len):
        super().__init__()
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.max_len = max_len
        self.pad_id = 0

    def __len__(self):
        return len(self.src_tokens)
    
    def __getitem__(self, index):
        src_data = self.src_tokens[index] + [self.pad_id] * (self.max_len - len(self.src_tokens[index])) # post zero padding
        tgt_data = self.tgt_tokens[index] + [self.pad_id] * (self.max_len - len(self.tgt_tokens[index])) # post zero padding
        return torch.LongTensor(src_data), torch.LongTensor(tgt_data)