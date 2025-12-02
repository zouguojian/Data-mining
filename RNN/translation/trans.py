# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import random
import math
import time
from collections import Counter
import spacy
import jieba   # pip install jieba

# 如果没有 spacy 英文模型，运行一次会自动下载
spacy_en = spacy.load("en_core_web_sm")
# 中文用 jieba 分词

# ==================== 1. 数据准备 ====================
# 下载 IWSLT 中英数据集（torchtext 内置）
from torchtext.datasets import IWSLT

train_iter = IWSLT(split='train', language_pair=('zh', 'en'))
valid_iter = IWSLT(split='valid', language_pair=('zh', 'en'))
test_iter  = IWSLT(split='test',  language_pair=('zh', 'en'))

# 构建中英文词汇表
SRC_VOCAB_SIZE = 15000   # 中文
TGT_VOCAB_SIZE = 10000   # 英文

def tokenize_zh(text):
    return list(jieba.cut(text))

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# 统计词频
src_counter = Counter()
tgt_counter = Counter()

print("正在统计词频...")
for src, tgt in train_iter:
    src_counter.update(tokenize_zh(src))
    tgt_counter.update(tokenize_en(tgt))

# 构建词汇表
src_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + \
            [w for w, c in src_counter.most_common(SRC_VOCAB_SIZE)]
tgt_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + \
            [w for w, c in tgt_counter.most_common(TGT_VOCAB_SIZE)]

src_word2idx = {w: i for i, w in enumerate(src_vocab)}
tgt_word2idx = {w: i for i, w in enumerate(tgt_vocab)}
src_idx2word = {i: w for w, i in src_word2idx.items()}
tgt_idx2word = {i: w for w, i in tgt_word2idx.items()}

# 句子转索引
def src_to_indices(sentence):
    return [src_word2idx.get(w, src_word2idx['<unk>']) for w in tokenize_zh(sentence)] + [src_word2idx['<eos>']]

def tgt_to_indices(sentence):
    return [tgt_word2idx['<sos>']] + \
           [tgt_word2idx.get(w, tgt_word2idx['<unk>']) for w in tokenize_en(sentence)] + \
           [tgt_word2idx['<eos>']]

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, data_iter):
        self.pairs = []
        for src, tgt in data_iter:
            src_ids = src_to_indices(src)
            tgt_ids = tgt_to_indices(tgt)
            if len(src_ids) <= 100 and len(tgt_ids) <= 100:  # 过滤超长句
                self.pairs.append((src_ids, tgt_ids))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    srcs = [item[0] for item in batch]
    tgts = [item[1] for item in batch]
    src_lens = torch.tensor([len(s) for s in srcs])
    tgt_lens = torch.tensor([len(t) for t in tgts])
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=src_word2idx['<pad>'])
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=tgt_word2idx['<pad>'])
    return src_pad, tgt_pad, src_lens, tgt_lens

BATCH_SIZE = 64
train_dataset = TranslationDataset(train_iter)
valid_dataset = TranslationDataset(valid_iter)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==================== 2. 模型：Encoder + Decoder + Luong Attention ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # 双向 → 单向 hidden

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hn, cn) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        # 合并双向 hidden
        hn = torch.cat((hn[-2], hn[-1]), dim=-1)
        cn = torch.cat((cn[-2], cn[-1]), dim=-1)
        hn = self.fc(hn)
        cn = self.fc(cn)
        return outputs, (hn.unsqueeze(0).contiguous(), cn.unsqueeze(0).contiguous())

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, B, H)
        # encoder_outputs: (B, L, H*2) → 先降维
        energy = torch.tanh(self.W(encoder_outputs))           # (B, L, H)
        attn_scores = torch.bmm(energy, decoder_hidden.squeeze(0).transpose(0,1))  # (B, L)
        attn_weights = torch.softmax(attn_scores, dim=1)       # (B, L)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H*2)
        return context, attn_weights

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = LuongAttention(hidden_dim)
        self.rnn = nn.LSTM(embed_dim + hidden_dim*2, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: (B,) → (B,1)
        embedded = self.embedding(input_token.unsqueeze(1))      # (B,1,E)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded.squeeze(1), context), dim=1).unsqueeze(1)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.out(output.squeeze(1))
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = len(tgt_vocab)

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(device)
        encoder_outputs, (hidden, cell) = self.encoder(src, src_len)

        input_token = tgt[:, 0]  # <sos>
        for t in range(1, tgt_len):
            pred, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = pred
            best_guess = pred.argmax(1)
            input_token = tgt[:, t] if random.random() < teacher_forcing_ratio else best_guess
        return outputs

# 初始化模型
HIDDEN_DIM = 512
EMBED_DIM  = 256
N_LAYERS   = 2
DROPOUT    = 0.5

encoder = EncoderRNN(len(src_vocab), EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
decoder = DecoderRNN(len(tgt_vocab), EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_word2idx['<pad>'])

# ==================== 3. 训练 ====================
def train_one_epoch():
    model.train()
    total_loss = 0
    for src, tgt, src_len, tgt_len in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, src_len, tgt, teacher_forcing_ratio=0.5)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, src_len, tgt_len in valid_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, src_len, tgt, teacher_forcing_ratio=0.0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            total_loss += loss.item()
    return total_loss / len(valid_loader)

# 训练
EPOCHS = 15
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()
    val_loss = evaluate()
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ==================== 4. 翻译测试（Greedy Decoding） ====================
def translate_sentence(sentence):
    model.eval()
    src_ids = torch.tensor([src_to_indices(sentence)], dtype=torch.long).to(device)
    src_len = torch.tensor([len(src_ids[0])])
    
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = encoder(src_ids, src_len)
        
        input_token = torch.tensor([tgt_word2idx['<sos>']]).to(device)
        translated = []
        
        for _ in range(100):  # 最大长度
            pred, hidden, cell, _ = decoder(input_token, hidden, cell, encoder_outputs)
            token_id = pred.argmax(1).item()
            if token_id == tgt_word2idx['<eos>']:
                break
            translated.append(tgt_idx2word[token_id])
            input_token = torch.tensor([token_id]).to(device)
        
        return " ".join(translated)

# 测试翻译
test_sentences = [
    "今天天气真好。",
    "我喜欢吃中餐。",
    "你好，世界！",
    "这是一个非常重要的会议。"
]

print("\n=== 中 → 英 翻译结果 ===")
for sent in test_sentences:
    trans = translate_sentence(sent)
    print(f"{sent} → {trans}")