from torch.utils.data import Dataset
import pandas as pd
import logging


class SentimentDataset(Dataset):
    def __init__(self, dataset_path=None, data=None, validation=0.0):
        super(SentimentDataset, self).__init__()
        if dataset_path is not None:
            self.dataset = pd.read_csv(dataset_path).values.tolist()
        else:
            self.dataset = data
        logging.info('Dataset Created')

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
