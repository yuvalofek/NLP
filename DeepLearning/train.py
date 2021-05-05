import argparse
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import SentimentDataset
from prepro import Preprocessor
from model import SentimentModel


def train(training, model, validation=None, optimizer=None, loss=torch.nn.BCELoss(), epochs=20):
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=0.01)
    train_len = len(training)
    for epoch in range(epochs):
        epoch_loss = 0.0
        with tqdm(enumerate(training), total=train_len, position=0) as t_epoch:
            t_epoch.set_description("Epoch {:02}/{}".format(epoch+1, epochs))
            for idx, (tweet, label) in t_epoch:
                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()

                # forward pass
                prediction = model(tweet)
                # calculate loss
                lss = loss(prediction.squeeze(), label.float())
                # backward + optimize
                lss.backward()
                optimizer.step()

                # running sum
                epoch_loss += lss.item()
                postfix = f'Training loss: {round(epoch_loss / (idx + 1), 4)}'
                t_epoch.set_postfix_str(postfix)
                break

        if validation is not None:
            test(validation, model, loss)


def test(testing, model, loss=torch.nn.BCELoss()):
    test_len = len(testing)
    epoch_loss = 0.0
    with tqdm(enumerate(testing), total=test_len, position=0) as t_epoch:
        t_epoch.set_description("Validation ")
        for idx, (tweet, label) in t_epoch:
            # forward pass
            prediction = model(tweet)
            # calculate loss
            lss = loss(prediction.squeeze(), label.float())

            # running sum
            epoch_loss += lss.item()
            postfix = f'Validation loss: {round(epoch_loss / (idx + 1), 4)}'
            t_epoch.set_postfix_str(postfix)


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
    print('Extracting dataset...')
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

    print('Initializing model...')
    mod = SentimentModel(len(preprocessor.vocab2enc)+3, args.embedding_dim, args.hidden_dim, args.batch_size)
    opt = Adam(mod.parameters(), lr=0.01)

    print('Training...')
    train(training=train_set, model=mod, validation=val_set, optimizer=opt, loss=torch.nn.BCELoss())



