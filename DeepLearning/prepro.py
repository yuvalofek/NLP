from collections import Counter
from data import SentimentDataset
import json


class Preprocessor:
    def __init__(self, max_vocab):
        self.max_vocab = max_vocab
        self.vocab2enc = None
        self.enc2vocab = None

    def fit(self, dataset):
        words = list()
        for i in range(len(dataset)):
            item = dataset.getitem(i)
            if item[1] is not None:
                words.extend(item[1].split(' '))
        vocab = Counter(words).most_common(self.max_vocab)
        self.vocab2enc = {word: i+1 for i, (word, _) in enumerate(vocab)}
        self.enc2vocab = {i+1: word for i, (word, _) in enumerate(vocab)}

        self.enc2vocab[0] = ''
        self.enc2vocab[self.max_vocab+2] = 'OOV'

    def encode(self, dataset):
        encoded = list()
        for i in range(len(dataset)):
            item = dataset.getitem(i)
            encoding = list()
            for word in item[1].split(' '):
                encoding.append(self.vocab2enc.get(word, self.max_vocab+2))
            encoded.append(list([item[0], encoding]))
        return SentimentDataset(data=encoded, data_from_file=False)

    def decode(self, dataset):
        encoded = list()
        for i in range(len(dataset)):
            item = dataset.getitem(i)
            encoding = list()
            for word in item[1]:
                encoding.append(self.enc2vocab.get(word, 'NAN'))
            encoded.append(list([item[0], ' '.join(encoding).strip()]))
        return SentimentDataset(data=encoded, data_from_file=False)

    @staticmethod
    def pad(dataset):
        max_len = 0
        for i in range(len(dataset)):
            item = dataset.getitem(i)
            if len(item[1]) > max_len:
                max_len = len(item[1])
        padded_data = list()
        for i in range(len(dataset)):
            item = dataset.getitem(i)
            padded_data.append([item[0], item[1].extend([0 for _ in range(max_len-len(item[1]))])])
        return SentimentDataset(data=padded_data, data_from_file=False)

    def transform(self, dataset):
        dataset = self.encode(dataset)
        self.pad(dataset)
        return dataset

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def save(self, file_name='./prepro_vocab.json'):
        with open(file_name, 'w') as f_out:
            json.dump({
                'vocab2enc': self.vocab2enc,
                'enc2vocab': self.enc2vocab,
            }, f_out)

    def load(self, file_name='./prepro_vocab.json'):
        with open(file_name, 'r') as f_in:
            data = json.load(f_in)

        self.vocab2enc = data['vocab2enc']
        self.enc2vocab = data['enc2vocab']


# if __name__ == '__main__':
#     p = Preprocessor(500)
#     s = SentimentDataset(data='./train.csv')
#     p.fit(s)
#
#     s_e = p.encode(s)
#     p.pad(s_e)
#     s_d = p.decode(s_e)
#
#     idx = 2
#     print(s.getitem(idx))
#     print(s_e.getitem(idx))
#     print(s_d.getitem(idx))
