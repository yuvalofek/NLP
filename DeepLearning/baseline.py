import argparse
import numpy as np

from data import SentimentDataset


def get_args():
    """
    Parse flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='./train.csv', help='training file path')
    return parser.parse_args()


if __name__ == '__main__':
    # Get arguments
    print('Getting arguments...')
    args = get_args()

    # make a dataset
    print('Importing dataset...')
    data = SentimentDataset(data=args.test_path)

    labels = [item[0] for item in data.data]
    print(f'Baseline Accuracy: {np.round(np.mean(labels), 4)*100}%')
