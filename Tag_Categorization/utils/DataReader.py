import pandas as pd


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
