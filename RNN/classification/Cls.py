# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import re
from collections import Counter
from torchtext.datasets import IMDB   # torchtext >= 0.6

# ==================== 1. 数据预处理 ====================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

# 构建词汇表
VOCAB_SIZE = 25000
MIN_FREQ = 5

print("正在构建词汇表...")
word_counts = Counter()
for label, text in IMDB(split='train'):
    word_counts.update(preprocess_text(text))

vocab = ['<pad>', '<unk>'] + [w for w, c in word_counts.most_common(VOCAB_SIZE) if c >= MIN_FREQ]
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")

def text_to_indices(text):
    return [word2idx.get(w, word2idx['<unk>']) for w in preprocess_text(text)]

# 自定义 Dataset
class IMDBDataset(Dataset):
    def __init__(self, split='train'):
        self.data = []
        for label_str, text in IMDB(split=split):
            label = 1 if label_str == 'pos' else 0
            seq = text_to_indices(text)
            self.data.append((seq, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def collate_fn(batch):
    seqs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=word2idx['<pad>'])
    return padded, labels, lengths

BATCH_SIZE = 64
train_dataset = IMDBDataset('train')
test_dataset  = IMDBDataset('test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==================== 2. 传统 RNN 模型（Vanilla RNN） ====================
class VanillaRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1,
                 bidirectional=True, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional,
                          nonlinearity='tanh')   # 传统 RNN 用 tanh
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        
    def forward(self, x, lengths):
        # x: (B, L)
        embedded = self.embedding(x)                     # (B, L, E)
        
        # pack 加速 + 正确处理 padding
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hn = self.rnn(packed)                         # hn: (num_layers*dir, B, H)
        
        # 取出最后一层隐状态
        if self.rnn.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=-1)     # (B, H*2)
        else:
            hn = hn[-1]                                  # (B, H)
            
        hn = self.dropout(hn)
        out = self.fc(hn).squeeze(1)                     # (B,)
        return out

# ==================== 3. 训练设置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = VanillaRNNClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=128,      # 传统 RNN 不宜太大
    num_layers=1,        # 强烈建议先用 1 层！2层以上很难收敛
    bidirectional=True,
    dropout=0.5
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==================== 4. 训练 & 评估函数 ====================
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for texts, labels, lengths in train_loader:
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        pred = (torch.sigmoid(outputs) > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate():
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for texts, labels, lengths in test_loader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            pred = (torch.sigmoid(outputs) > 0.5).float()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ==================== 5. 开始训练 ====================
EPOCHS = 15
print("开始训练（传统 Vanilla RNN）...")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# ==================== 6. 单句预测函数 ====================
def predict(text):
    model.eval()
    indices = text_to_indices(text)
    seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    length = torch.tensor([len(indices)], dtype=torch.long).to(device)
    with torch.no_grad():
        logit = model(seq, length)
        prob = torch.sigmoid(logit).item()
    return "正面" if prob > 0.5 else "负面", prob

# 测试几条评论
test_reviews = [
    "This movie is absolutely fantastic! I loved every minute of it.",
    "Worst film ever. Complete waste of time and money.",
    "It's okay, not great but not terrible either.",
    "An absolute masterpiece with brilliant acting and direction."
]

print("\n=== 单句预测 ===")
for rev in test_reviews:
    sentiment, prob = predict(rev)
    print(f"{sentiment} ({prob:.4f}) → {rev}")