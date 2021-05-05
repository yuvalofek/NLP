import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import SentimentDataset
from prepro import Preprocessor
from model import SentimentModel


def binary_cross_entropy(true, prediction):
    return true*torch.log(prediction)+prediction*torch.log(true)


def train(training, model, optimizer, loss):
    epoch_loss = 0.0
    for idx, (tweet, label) in enumerate(training):
        # get the true values & make into cuda if needed

        # tweet = torch.tensor(tweet)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        prediction = model(tweet)
        # calculate multi-loss
        lss = loss(prediction.squeeze(), label.float())
        # backward + optimize
        lss.backward()
        optimizer.step()

        # running sum
        epoch_loss += lss.item()
        postfix = f'Training loss: {round(epoch_loss / (idx + 1), 4)}'


def get_args():
    """
    Parse flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./train.csv', help='training file path')
    parser.add_argument('--validation_path', type=str, default=None, help='validation file path')
    parser.add_argument('--batch_size', type=int, default=16, help='training set batch size')
    parser.add_argument('--max_vocab', type=int, default=10_000, help='maximum vocab size')
    parser.add_argument('--embedding_dim', type=int, default=6, help='size of embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=6, help='size of hidden layer')
    parser.add_argument('--save_path', type=str, default='./output.labels', help='file path for saved_model')
    return parser.parse_args()


if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # make a dataset
    data = SentimentDataset(data=args.train_path)

    # preprocess and save word encodings
    preprocessor = Preprocessor(max_vocab=args.max_vocab)
    data = preprocessor.fit_transform(dataset=data)
    preprocessor.save()

    # validation split
    data.split_data(validation_count=10_000)
    train_ds, val_ds = data.to_dataset()

    # to dataLoaders
    train_set = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_set = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    mod = SentimentModel(len(preprocessor.vocab2enc)+3, args.embedding_dim, args.hidden_dim, args.batch_size)
    opt = Adam(mod.parameters(), lr=0.01)

    train(train_set, mod, opt, torch.nn.BCELoss())



