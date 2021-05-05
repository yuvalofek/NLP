from torch.utils.data import Dataset
import pandas as pd


class SentimentDataset(Dataset):
    def __init__(self, dataset_path='./train.csv'):
        super(SentimentDataset, self).__init__()
        self.dataset = pd.read_csv(dataset_path).values.tolist()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
