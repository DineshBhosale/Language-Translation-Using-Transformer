import torch
import pandas as pd
from transformer import *
from dataloading import *
from timeit import default_timer as timer

device = torch.device('mps')
torch.manual_seed(1337)
torch.mps.set_per_process_memory_fraction(0.0)

batch_size = 512
src_tokens = []
tgt_tokens = []
src_stoi_vocab = {'<pad>': 0, '[start]': 1, '[end]': 2} 
src_itos_vocab = {0:'<pad>', 1:'[start]', 2:'[end]'}
tgt_stoi_vocab = {'<pad>': 0, '[start]': 1, '[end]': 2}
tgt_itos_vocab = {0:'<pad>', 1:'[start]', 2:'[end]'}

df_src = pd.read_csv("./small_vocab_en.csv", sep='\t', header = None)
df_src = df_src.rename(columns={0:"src"})

df_tgt = pd.read_csv('./small_vocab_fr.csv', sep='\t', header = None)
df_tgt = df_tgt.rename(columns={0:"tgt"})

df = pd.concat([df_src, df_tgt], axis=1)
df = df.sample(frac=1, random_state=42)
print("Sample Data")
print("Source Language\t\t\t\t\t\t\t\tTarget Language")

print_sample_data(df, sample=5)

print("Total Number of Sentences: {0}".format(len(df_src)))

df['src'] = df['src'].apply(lambda text: clean_prepare_text(text))
df['tgt'] = df['tgt'].apply(lambda text: clean_prepare_text(text))

df['src'].apply(lambda text: create_vocab(text, src_stoi_vocab, src_itos_vocab))
df['tgt'].apply(lambda text: create_vocab(text, tgt_stoi_vocab, tgt_itos_vocab))

src_vocab_size, tgt_vocab_size = len(src_stoi_vocab), len(tgt_stoi_vocab)
print("Source Vocab Size is {0} and Target Vocab Size is {1}".format(src_vocab_size, tgt_vocab_size))

# encode sentences into tokens
print("Encoding Sentences into tokens")
for src_sent, tgt_sent in zip(df['src'], df['tgt']):
    src_tokens.append(tokenize_sentence(src_sent, src_stoi_vocab))
    tgt_tokens.append(tokenize_sentence(tgt_sent, tgt_stoi_vocab))

max_seq_length = max([max(len(src_token), len(tgt_token)) for src_token, tgt_token in zip(src_tokens, tgt_tokens)])
print("Maximum Sequence Length: {}".format(max_seq_length))

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

model = Transformer(embd_sze=512, src_vocab_sze=src_vocab_size, tgt_vocab_sze=tgt_vocab_size, max_seq_len=max_seq_length)
model.to(device)
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))

model.load_state_dict(torch.load("./model.pt", weights_only=True))
model.to(device)
model.eval()
for i, (src, tgt) in enumerate(test_loader):
    src, tgt = src.to(device), tgt

    tgt_input = torch.LongTensor([src_stoi_vocab['<pad>']] * max_seq_length).to(device)
    tgt_input[0] = tgt_stoi_vocab['[start]']
    tgt_input =  tgt_input.reshape(1, -1)

    src_mask, tgt_mask = make_mask(src, tgt_input, src_stoi_vocab, tgt_stoi_vocab, device)
    e_output = model.encode(src, src_mask)
    op = [tgt_stoi_vocab['[start]']]
    for j in range(max_seq_length-1):
        src_mask, tgt_mask = make_mask(src, tgt_input, src_stoi_vocab, tgt_stoi_vocab, device)
        output = model.decode(tgt_input, e_output, src_mask, tgt_mask)
        output = F.softmax(output, dim=-1)
        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][j].item()
        #print(last_word_id)
        if last_word_id == 0:
            continue
        tgt_input[0][j+1] = last_word_id

        #print(next_word)
        #probs = F.softmax(output, dim=-1)
        #idx_next = torch.argmax(probs, dim=-1)

        # if idx_next[0][j].item() == tgt_stoi_vocab['[end]']:
        #     break

        #tgt_input[0][j+1] = next_word
        op.append(last_word_id)
        
    print(' '.join(src_itos_vocab[o] for o in src.tolist()[0] if src_itos_vocab[o] != '<pad>' ) + "\t\t\t\t" + ' '.join(tgt_itos_vocab[o] for o in op))