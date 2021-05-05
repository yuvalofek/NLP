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
        # to separate arrays
        train_x, train_y = [item[1] for item in self.data], [item[0] for item in self.data]
        train_x = np.array(train_x)
        tr_ds = TensorDataset(torch.tensor(train_x, dtype=torch.long), torch.FloatTensor(train_y))

        val_ds = None
        if self.validation is not None:
            val_x, val_y = [item[1] for item in self.validation], [item[0] for item in self.validation]
            val_x = np.array(val_x)
            val_ds = TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y))

        return tr_ds, val_ds
