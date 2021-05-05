from torch.nn import Module
import torch.nn as nn
from torch import sigmoid


class SentimentModel(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(SentimentModel, self).__init__()
        self.n_layers = 2
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, 1)
        self.sigmoid = sigmoid

    def forward(self, x):
        batch_size = x.size()
        # embedding
        embeds = self.word_embedding(x)

        # lstm
        lstm_out, _ = self.lstm(embeds)
        # lstm_out, _ = lstm_out.view(-1, self.hidden_dim)

        # linear to sigmoid output
        out = self.l1(lstm_out)
        prediction = self.sigmoid(out)
        prediction = prediction.view(batch_size, -1)
        prediction = prediction[:, -1]

        return prediction
