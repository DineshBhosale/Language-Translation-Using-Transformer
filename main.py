import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.transformer import *
from utils.dataloading import *
from timeit import default_timer as timer

device = torch.device('mps')

def set_seeds():
    # fix random seeds
    SEED_VALUE = 42
 
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
set_seeds()

# torch.manual_seed(1337)
# torch.mps.set_per_process_memory_fraction(0.0)

batch_size = 512
src_tokens = []
tgt_tokens = []

df_src = pd.read_csv("./data/small_vocab_en.csv", sep='\t', header = None)
df_src = df_src.rename(columns={0:"src"})

df_tgt = pd.read_csv('./data/small_vocab_fr.csv', sep='\t', header = None)
df_tgt = df_tgt.rename(columns={0:"tgt"})

df = pd.concat([df_src, df_tgt], axis=1)
df = df.sample(frac=1, random_state=42)

print("Displaying Sample Data")
print("\t\t\t\t\t\t\t\tSource Language\t\t\t\t\t\t\t\t\t\t\t\t\t\tTarget Language")
print_sample_data(df, sample=5)
print("\nTotal Number of Sentences: {0}\n".format(len(df_src)))

df['src'] = df['src'].apply(lambda text: clean_prepare_text(text))
df['tgt'] = df['tgt'].apply(lambda text: clean_prepare_text(text))

vocab_s = Vocabulary("src")
vocab_s.generate_vocabulary(df['src'].values)
vocab_s.save_vocab()

vocab_t = Vocabulary("tgt")
vocab_t.generate_vocabulary(df['tgt'].values)
vocab_t.save_vocab()

src_vocab_size, tgt_vocab_size = len(vocab_s.stoi), len(vocab_t.stoi)
print("Source Vocab Size is {0} and Target Vocab Size is {1}".format(src_vocab_size, tgt_vocab_size))

# encode sentences into tokens
for src_sent, tgt_sent in tqdm(zip(df['src'], df['tgt']), total=len(df['src']), desc="Encoding Sentences into tokens"):
    src_tokens.append(vocab_s.encode_str(src_sent))
    tgt_tokens.append(vocab_t.encode_str(tgt_sent))

max_seq_length = max([max(len(src_token), len(tgt_token)) for src_token, tgt_token in zip(src_tokens, tgt_tokens)])
print("\nMaximum Sequence Length: {}\n".format(max_seq_length))

# create train-val-test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

num_sentences = len(df)
num_train = int(train_ratio * num_sentences)
num_val = int(val_ratio * num_sentences)
num_test = num_sentences - num_train - num_val

train_src_tokens, train_tgt_tokens = src_tokens[:num_train], tgt_tokens[:num_train]
val_src_tokens, val_tgt_tokens = src_tokens[num_train:num_train+num_val], tgt_tokens[num_train:num_train+num_val]
test_src_tokens, test_tgt_tokens = src_tokens[num_train+num_val:], tgt_tokens[num_train+num_val:]

train_dataset = TranslationDataset(train_src_tokens, train_tgt_tokens, max_len=max_seq_length)
val_dataset = TranslationDataset(val_src_tokens, val_tgt_tokens, max_len=max_seq_length)
test_dataset = TranslationDataset(test_src_tokens, test_tgt_tokens, max_len=max_seq_length)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

print("Loading Model")
model = Transformer(embd_sze=512, src_vocab_sze=src_vocab_size, tgt_vocab_sze=tgt_vocab_size, max_seq_len=max_seq_length)
model.to(device)
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))

model.load_state_dict(torch.load("./weights/model-iteration-02.pt", weights_only=True))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.NLLLoss().to(device)
epochs = 5

# model training
for epoch in range(1, epochs+1):
    start_time = timer()
    train_loss = 0
    model.train()
    # train loop
    for i, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = vocab_s.get_mask(src)
        tgt_mask = vocab_t.get_mask(tgt_input, device)

        # zero out gradients before every batch
        optimizer.zero_grad()

        # forward pass
        e_output = model.encode(src, src_mask)
        output = model.decode(tgt_input, e_output, src_mask, tgt_mask)

        # calculate loss
        loss = loss_func(output.view(-1, output.size(-1)), tgt_output.reshape(tgt_output.shape[0] * tgt_output.shape[1]))
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
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = vocab_s.get_mask(src)
            tgt_mask = vocab_t.get_mask(tgt_input, device)

            # zero out gradients before every batch
            optimizer.zero_grad()

            # forward pass
            e_output = model.encode(src, src_mask)
            output = model.decode(tgt_input, e_output, src_mask, tgt_mask)

            # calculate loss
            loss = loss_func(output.view(-1, output.size(-1)), tgt_output.reshape(tgt_output.shape[0] * tgt_output.shape[1]))
            
            val_loss += loss.item()
    end_time = timer()

    print("Epoch: {0} | Training Loss: {1} | Validation Loss: {2} | Epoch Time: {3}s".\
          format(epoch, round(train_loss/len(train_loader), 4), round(val_loss / len(val_loader), 4), round(end_time-start_time, 4)))

torch.save(model.state_dict(), "./weights/model-iteration-03.pt")