import re
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformer import *
import torch.utils.data as data
from unicodedata import normalize
from torch.nn import functional as F

device = torch.device('mps')
torch.manual_seed(1337)
torch.mps.set_per_process_memory_fraction(0.0)

def clean_text(text):
    text = normalize('NFD', text.lower())
    text = re.sub('[^A-Za-z ]+', '', text)
    return text

def clean_prepare_text(text):
    text = '[start] ' + clean_text(text) + ' [end]'
    return text

def make_mask(src, tgt):
    src_mask = (src != src_vocab["<pad>"]).unsqueeze(1)
    tgt_mask = (tgt != tgt_vocab["<pad>"]).unsqueeze(1)
    
    dec_attention_mask = torch.tril(torch.ones((1, tgt.shape[1], tgt.shape[1]), dtype=torch.bool)).to(device)
    tgt_mask = tgt_mask & dec_attention_mask
    return src_mask, tgt_mask

# Read Data
df = pd.read_csv('./en-fr.txt', names = ['en', 'fr', 'attr'], usecols=['en', 'fr'], sep='\t')
print("Number of Sentences: {}".format(len(df)))
df = df.sample(frac=1, random_state=42)
df = df.reset_index(drop=True)
df.head()

df['en'] = df['en'].apply(lambda text: clean_text(text))
df['fr'] = df['fr'].apply(lambda text: clean_prepare_text(text))

# create train-val-test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

num_sentences = len(df)
num_train = int(train_ratio * num_sentences)
num_val = int(val_ratio * num_sentences)
num_test = num_sentences - num_train - num_val

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

train_df = df[:num_train]
val_df = df[num_train:num_train+num_val]
test_df = df[num_train+num_val:]

train_df.head(1)

# tokenization
train_src_tokens = []
train_tgt_tokens = []
val_src_tokens = []
val_tgt_tokens = []
test_src_tokens = []
test_tgt_tokens = []
src_vocab = {'<pad>': 0}  # Initialize source vocabulary with <pad> token
tgt_vocab = {'<pad>': 0}  # Initialize target vocabulary with <pad> token

src_dec_vocab = {0:'<pad>'}
tgt_dec_vocab = {0:'<pad>'}

def tokenize_sentence(sentence, enc_vocab, dec_vocab):
    tokens = sentence.split()
    token_ids = []
    for token in tokens:
        if token not in enc_vocab:
            enc = len(enc_vocab)
            enc_vocab[token] = enc
            dec_vocab[enc] = token
        token_ids.append(enc_vocab[token])
    return token_ids

# Tokenizing training data
print("Tokenizing training data of length: {}".format(len(train_df)))
for src_sent, tgt_sent in zip(train_df['en'], train_df['fr']):
    train_src_tokens.append(tokenize_sentence(src_sent, src_vocab, src_dec_vocab))
    train_tgt_tokens.append(tokenize_sentence(tgt_sent, tgt_vocab, tgt_dec_vocab))

# Tokenizing validation data
print("Tokenizing validation data of length: {}".format(len(val_df)))
for src_sent, tgt_sent in zip(val_df['en'], val_df['fr']):
    val_src_tokens.append(tokenize_sentence(src_sent, src_vocab, src_dec_vocab))
    val_tgt_tokens.append(tokenize_sentence(tgt_sent, tgt_vocab, tgt_dec_vocab))

# Tokenizing testing data
print("Tokenizing testing data of length: {}".format(len(test_df)))
for src_sent, tgt_sent in zip(test_df['en'], test_df['fr']):
    test_src_tokens.append(tokenize_sentence(src_sent, src_vocab, src_dec_vocab))
    test_tgt_tokens.append(tokenize_sentence(tgt_sent, tgt_vocab, tgt_dec_vocab))

# Update the vocabulary sizes
src_vocab_size, tgt_vocab_size = len(src_vocab), len(tgt_vocab)
print("Source Vocab Size is {0} and Target Vocab Size is {1}".format(src_vocab_size, tgt_vocab_size))

# get max length for padding
max_len = 16 

class TranslationDataset(data.Dataset):
    def __init__(self, src_tokens, tgt_tokens):
        super().__init__()
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.max_len = 16
        #self.max_len = max([max(len(src_token), len(tgt_token)) for src_token, tgt_token in zip(src_tokens, tgt_tokens)]) # extract maximum length for zero padding

    def __len__(self):
        return len(self.src_tokens)
    
    def __getitem__(self, index):
        src_data = self.src_tokens[index] + [0] * (self.max_len - len(self.src_tokens[index])) # post zero padding
        tgt_data = self.tgt_tokens[index] + [0] * (self.max_len - len(self.tgt_tokens[index])) # post zero padding

        return torch.tensor(src_data), torch.tensor(tgt_data)

train_dataset = TranslationDataset(train_src_tokens, train_tgt_tokens)
val_dataset = TranslationDataset(val_src_tokens, val_tgt_tokens)
test_dataset = TranslationDataset(test_src_tokens, test_tgt_tokens)

batch_size = 512
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

max_seq_length = max(train_dataset.max_len, val_dataset.max_len, test_dataset.max_len)  # Maximum sequence length
model = Transformer(embd_sze=512, src_vocab_sze=src_vocab_size, tgt_vocab_sze=tgt_vocab_size, max_seq_len=max_seq_length)
model.to(device)
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = torch.nn.CrossEntropyLoss().to(device)
epochs = 20
# model training
for epoch in range(1, epochs+1):
    train_loss = 0
    model.train()
    # train loop
    for i, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = make_mask(src, tgt)

        # zero out gradients before every batch
        optimizer.zero_grad()

        # forward pass
        output = model(src, tgt, src_mask, tgt_mask)

        # calculate loss
        loss = loss_func(output.view(-1, output.size(-1)), tgt.view(tgt.shape[0] * tgt.shape[1]))
        # backward pass
        loss.backward()

        # gradient descent
        optimizer.step()
        train_loss += loss.item()

    # valdition loop
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (src, tgt) in enumerate(val_loader):
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = make_mask(src, tgt)

            # forward pass
            output = model(src, tgt, src_mask, tgt_mask)
            # calculate loss
            loss = loss_func(output.view(-1, output.size(-1)), tgt.view(tgt.shape[0] * tgt.shape[1]))
            
            val_loss += loss.item()

    print("Epoch: {0} | Training Loss: {1} | Validation Loss: {2}".format(epoch, round(train_loss/len(train_loader), 4), round(val_loss / len(val_loader), 4)))

torch.save(model.state_dict(), "./model.pt")

# model inference
def translate(sentence):
    tokenized_src_sentence = tokenize_sentence(sentence, src_vocab, src_dec_vocab)
    src = torch.tensor(tokenized_src_sentence + [0] * (max_seq_length- len(tokenized_src_sentence)))
    src =  src.reshape(1, -1).to(device)
    dec_sen = " "
    for i in range(max_seq_length):

        tokenized_tgt_sentence = tokenize_sentence("[start]", tgt_vocab, tgt_dec_vocab)
        tgt = torch.tensor(tokenized_tgt_sentence + [0] * (max_seq_length- len(tokenized_tgt_sentence)))
        tgt = tgt.reshape(1, -1).to(device)

        src_mask, tgt_mask = make_mask(src, tgt)
        output = model(src, tgt, src_mask, tgt_mask)
        
        #output = torch.multinomial(F.softmax(output[:, -1, :], dim=-1), num_samples=1)
        output = F.softmax(output, dim=-1)
        output = torch.argmax(output, dim=-1)
        dec_token = output[0][i].item()
        #dec_token = output[0].item()
        if dec_token == tgt_vocab['[end]']:
            return dec_sen
        if tgt_dec_vocab[dec_token] == '<pad>':
            continue

        dec_sen += tgt_dec_vocab[dec_token]
        tokenized_tgt_sentence.append(dec_token)
    
    return dec_sen
        
sentence = train_df['en'][10]
print(sentence)
dec_sen = translate(sentence)
print(dec_sen)

