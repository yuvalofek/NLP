import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data import SentimentDataset
from prepro import Preprocessor
from train import SentimentModel

def binary_crossentropy(true, pred):
    return true*torch.log(pred)


def train(tr, model, optimizer, loss):
    epoch_loss = 0.0
    for batch_i, batch in enumerate(training_set):
        # get the true values & make into cuda if needed
        true = [torch.tensor(enc) for enc in batch]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        pred = model(true)
        # calculate multi-loss
        lss = loss(true, pred)
        # backward + optimize
        lss.backward()
        optimizer.step()

        # running sum
        epoch_loss += lss.item()
        postfix = f'Training loss: {round(epoch_loss / (batch_i + 1), 4)}'


def get_args():
    """
    Parse flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./train.csv', help='training file path')
    parser.add_argument('--validation_path', type=str, default=None, help='validation file path')
    parser.add_argument('--batch_size', type=int, default=16, help='training set batch size')
    parser.add_argument('--max_vocab', type=int, default=10_000, help='maximum vocab size')
    parser.add_argument('--save_path', type=str, default='./output.labels', help='file path for saved_model')
    return parser.parse_args()


if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # make a dataset
    training_set = SentimentDataset(dataset_path=args.train_path)

    # preprocess and save word encodings
    preprocessor = Preprocessor(max_vocab=args.max_vocab)
    training_set = preprocessor.fit_transform(dataset=training_set)
    preprocessor.save()

    # training_set = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)




