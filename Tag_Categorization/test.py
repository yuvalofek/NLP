import pandas as pd
import numpy as np
import argparse

class DataReader:
    """
    For reading in the files and getting the paths and labels
    """

    def __init__(self, path):
        # reading in the paths and the labels
        df = pd.read_csv(path, delimiter=' ', names=['path', 'label'])
        # some parameters
        self.available_labels = list(df['label'].unique())
        self.paths = df['path'].to_list()
        self.labels = df['label'].to_list()

    def get_data(self):
        """
        return: paths and labels (if exist)
        """
        return self.paths, self.labels

def get_args():
    """
    Parse flags
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--f1', type=str, default='./output.labels')
    parser.add_argument('--f2', type=str, default='./corpus1_test.labels')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inp1 = DataReader(args.f1)
    inp2 = DataReader(args.f2)

    _, y1 = inp1.get_data()
    _, y2 = inp2.get_data()

    print(np.array([y == y2[i] for i, y in enumerate(y1)]).mean(

    ))
