from torch.nn import Module
import torch.nn as nn
from torch import sigmoid
import torch


class SentimentModel(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(SentimentModel, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, dropout=dropout)
        self.l1 = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        batch_size = x.size()
        # embedding
        embeds = self.word_embedding(x)

        # lstm
        lstm_out, _ = self.lstm(embeds)
        # lstm_out, _ = lstm_out.view(-1, self.hidden_dim)

        # fully connected and sigmoid output
        out = self.l1(lstm_out)
        prediction = sigmoid(out)
        prediction = prediction.view(batch_size, -1)
        prediction = prediction[:, -1]

        return prediction
