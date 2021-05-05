from torch.utils.data import Dataset
import pandas as pd
import logging


class SentimentDataset:
    def __init__(self, data, data_from_file=True):
        if data_from_file:
            self.data = pd.read_csv(data).values.tolist()
        else:
            self.data = data
        logging.info('Dataset Created')

    def getitem(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_data(self, validation_count):
        # training set, validation set
        return self.data[validation_count:], self.data[:validation_count]