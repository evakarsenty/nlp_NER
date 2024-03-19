from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import preprocessing as pp
from sklearn.metrics import f1_score

train_path = 'train.tagged'
val_path = 'dev.tagged'
test_path = 'test.untagged'


class LSTMDataSet(Dataset, ABC):

    def __init__(self, sentences, sentences_lens, y):
        self.X = sentences
        self.X_lens = sentences_lens
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.X_lens[item], self.y[item]


class LSTMDataSetTest(Dataset, ABC):

    def __init__(self, sentences, sentences_lens):
        self.X = sentences
        self.X_lens = sentences_lens

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.X_lens[item]


def tokenize(x_train, x_val, x_test):
    word2idx = {"[PAD]": 0, "[UNK]": 1}
    idx2word = ["[PAD]", "[UNK]"]
    for sent in x_train:
        for word in sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)

    final_list_train, final_list_test, final_list_val = [], [], []
    for sent in x_train:
        final_list_train.append([word2idx[word] for word in sent])
    for sent in x_val:
        final_list_val.append([word2idx[word] if word in word2idx else word2idx['[UNK]'] for word in sent])
    for sent in x_test:
        final_list_test.append([word2idx[word] if word in word2idx else word2idx['[UNK]'] for word in sent])

    return final_list_train, final_list_val, final_list_test, word2idx, idx2word


def padding_(sentences, seq_len, tag):
    if not tag:
        features = np.zeros((len(sentences), seq_len), dtype=int)
    else:
        features = np.full((len(sentences), seq_len), 0, dtype=int)

    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]

    return torch.tensor(features)


class MyLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim=30, hidden_dim=50, tag_dim=2, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.hidden2tag = nn.Sequential(nn.ReLU(), nn.Linear(self.hidden_dim, tag_dim))
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, sentence, sentence_lens, tags=None):
        embeds = self.word_embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)

        if tags is not None:
            loss = self.loss_fn(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1))
            return tag_scores, loss
        return tag_scores, None


def train_LSTM(model, device, optimizer, train_dataset, val_dataset):
    y_true = []
    y_pred = []

    for phase in ["train", "validation"]:
        if phase == "train":
            model.train(True)
        else:
            model.train(False)

        dataset = train_dataset if phase == "train" else val_dataset
        t_bar = tqdm(dataset)

        for sentence, lens, tags in t_bar:
            if phase == "train":
                model.zero_grad()
                tag_scores, loss = model(sentence.to(device), lens.to(device), tags.to(device).long())
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    tag_scores, _ = model(sentence.to(device), lens.to(device), tags.to(device).long())

            for tag in tags:
                y_true.extend(tag.cpu().numpy().flatten())
            for tag in tag_scores.argmax(2):
                y_pred.extend(tag.cpu().numpy().flatten())

        f1 = f1_score(y_true, y_pred)
        print(f"{phase} F1-score: {f1:.2f}")

    return f1


def test_LSTM(device, model, test_data_loader, output_file):
    predictions = []
    test_data = pp.get_sentences_and_tags(test_path, False)

    with torch.no_grad():
        for sentence, lens in test_data_loader:
            tag_scores, _ = model(sentence.to(device), lens.to(device))
            predictions.extend(tag_scores.argmax(2))

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, tags in zip(test_data, predictions):
            for word, tag in zip(sentence, tags):
                if tag == 0:
                    f.write(f"{word}\t{'O'}\n")
                else:
                    f.write(f"{word}\t{'1'}\n")
            f.write("\n")

