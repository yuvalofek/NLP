import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import logging


class SentimentDataset:
    def __init__(self, data, data_from_file=True):
        if data_from_file:
            self.data = pd.read_csv(data).values.tolist()
        else:
            self.data = data
        logging.info('Dataset Created')
        self.validation = None

    def getitem(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def split_data(self, validation_count):
        # training set, validation set
        self.validation = self.data[:validation_count]
        self.data = self.data[validation_count:]
        return self.data, self.validation

    def to_dataset(self):
        tr_ds = self.extract_from_list(self.data)
        val_ds = None
        if self.validation is not None:
            val_ds = self.extract_from_list(self.validation)
        return tr_ds, val_ds

    @staticmethod
    def extract_from_list(dataset):
        x, y = [item[1] for item in dataset], [item[0] for item in dataset]
        x = np.array(x)
        ds = TensorDataset(torch.tensor(x, dtype=torch.long), torch.FloatTensor(y))
        return ds
