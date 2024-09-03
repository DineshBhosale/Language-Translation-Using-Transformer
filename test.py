import torch
import pandas as pd
from utils.transformer import *
from utils.dataloading import *
from tqdm import tqdm
from timeit import default_timer as timer

device = torch.device('mps')
torch.manual_seed(1337)
torch.mps.set_per_process_memory_fraction(0.0)

src_tokens = []
tgt_tokens = []

df_src = pd.read_csv("./data/small_vocab_en.csv", sep='\t', header = None)
df_src = df_src.rename(columns={0:"src"})

df_tgt = pd.read_csv('./data/small_vocab_fr.csv', sep='\t', header = None)
df_tgt = df_tgt.rename(columns={0:"tgt"})

df = pd.concat([df_src, df_tgt], axis=1)
df = df.sample(frac=1, random_state=42)

print("Sample Data")
print("Source Language\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tTarget Language")
print_sample_data(df, sample=5)
print("Total Number of Sentences: {0}".format(len(df_src)))

df['src'] = df['src'].apply(lambda text: clean_prepare_text(text))
df['tgt'] = df['tgt'].apply(lambda text: clean_prepare_text(text))

vocab_s = Vocabulary("src")
vocab_s.generate_vocabulary(df['src'].values)

vocab_t = Vocabulary("tgt")
vocab_t.generate_vocabulary(df['tgt'].values)

src_vocab_size, tgt_vocab_size = len(vocab_s.stoi), len(vocab_t.stoi)
print("Source Vocab Size is {0} and Target Vocab Size is {1}\n".format(src_vocab_size, tgt_vocab_size))

# encode sentences into tokens
for src_sent, tgt_sent in tqdm(zip(df['src'], df['tgt']), total=len(df['src']), desc="Encoding Sentences into tokens"):
    src_tokens.append(vocab_s.encode_str(src_sent))
    tgt_tokens.append(vocab_t.encode_str(tgt_sent))

max_seq_length = max([max(len(src_token), len(tgt_token)) for src_token, tgt_token in zip(src_tokens, tgt_tokens)])
print("Maximum Sequence Length: {}\n".format(max_seq_length))

# create train-val-test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

num_sentences = len(df)
num_train = int(train_ratio * num_sentences)
num_val = int(val_ratio * num_sentences)
num_test = num_sentences - num_train - num_val

test_src_tokens, test_tgt_tokens = src_tokens[num_train+num_val:], tgt_tokens[num_train+num_val:]
test_dataset = TranslationDataset(test_src_tokens, test_tgt_tokens, max_len=max_seq_length)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

model = Transformer(embd_sze=512, src_vocab_sze=src_vocab_size, tgt_vocab_sze=tgt_vocab_size, max_seq_len=max_seq_length)
print("Number of parameters: {} M".format(sum(p.numel() for p in model.parameters())/1e6))

model.to(device)
model.eval()
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
    
for src, tgt in test_loader:
    src = src.to(device)
    src_mask = vocab_s.get_mask(src)
    e_output = model.encode(src, src_mask)
    d_output = inference(e_output, src_mask)

    print(vocab_s.decode_tokens(src.tolist()[0]) + '\t' + vocab_t.decode_tokens(d_output))