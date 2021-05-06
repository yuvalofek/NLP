import argparse
import torch

from torch.utils.data import DataLoader

from train import test
from data import SentimentDataset
from prepro import Preprocessor


def get_args():
    """
    Parse flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='./train.csv', help='training file path')
    parser.add_argument('--max_vocab', type=int, default=5_000, help='maximum vocab size')
    parser.add_argument('--model_path', type=str, default='./trained_model.pkl', help='path to trained model')
    parser.add_argument('--prepro_path', type=str, default='./prepro_vocab.json', help='path to fit preprocessor')
    return parser.parse_args()


if __name__ == '__main__':
    # Get arguments
    print('Getting arguments...')
    args = get_args()

    # make a dataset
    print('Importing dataset...')
    data = SentimentDataset(data=args.test_path)

    # preprocess and save word encodings

    preprocessor = Preprocessor(max_vocab=args.max_vocab)
    preprocessor.load()
    data = preprocessor.transform(dataset=data)

    # validation split
    test_ds, _ = data.to_dataset()

    # to dataLoaders
    test_set = DataLoader(test_ds, batch_size=16, shuffle=False)

    # load saved model
    print('Loading trained model...')
    model = torch.load(args.model_path)
    model.eval()

    test(test_set, model)
