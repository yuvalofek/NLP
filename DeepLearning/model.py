from torch.nn import Module
import torch.nn as nn
from torch.nn.functional import sigmoid


class SentimentModel(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        prediction = sigmoid(self.output(lstm_out.view(len(sentence), -1)))
        return prediction
