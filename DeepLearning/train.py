import argparse
import torch
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import SentimentDataset
from prepro import Preprocessor
from model import SentimentModel


def fit(training, model, validation=None, optimizer=None, loss=torch.nn.BCELoss(), epochs=20):
    # if no optimizer, set one
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=0.01)

    # fit the model!
    train_loss = []
    val_loss = []
    val_acc = []
    for epoch in range(epochs):
        # train
        tr_loss = train(training, model, optimizer, loss, epoch, epochs)
        train_loss.append(tr_loss)
        # validate
        if validation is not None:
            lss, acc = test(validation, model, loss)
            val_loss.append(lss)
            val_acc.append(acc)
    return train_loss if validation is None else (train_loss, val_loss, val_acc)


def train(training, model, optimizer, loss, epoch=0, epochs=1):
    epoch_loss = 0.0
    train_len = len(training)

    with tqdm(enumerate(training), total=train_len, position=0) as t_epoch:
        t_epoch.set_description("Epoch {:02}/{}".format(epoch + 1, epochs))
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
    return epoch_loss


def test(testing, model, loss=torch.nn.BCELoss()):
    test_len = len(testing)
    epoch_loss = 0.0
    acc_tot = 0.0
    with tqdm(enumerate(testing), total=test_len, position=0) as t_epoch:
        t_epoch.set_description("Validation ")
        for idx, (tweet, label) in t_epoch:
            # forward pass
            prediction = model(tweet)
            # calculate loss
            lss = loss(prediction.squeeze(), label.float())
            acc = np.mean(np.round(prediction.detach().numpy()) == label.detach().numpy())

            # running sum
            epoch_loss += lss.item()
            acc_tot += acc
            postfix = f'Loss: {round(epoch_loss / (idx + 1), 4)} Accuracy: {round(acc_tot / (idx + 1)*100, 3)}%'
            t_epoch.set_postfix_str(postfix)
    return epoch_loss, acc_tot


def get_args():
    """
    Parse flags
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./train.csv', help='training file path')
    parser.add_argument('--validation_count', type=int, default=5_000, help='number of inputs to save for validation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--max_vocab', type=int, default=5_000, help='maximum vocab size')
    parser.add_argument('--embedding_dim', type=int, default=8, help='embedding dimension size')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden layer size')
    parser.add_argument('--model_save_path', type=str, default='./trained_model.pkl', help='file path for saved model')
    parser.add_argument('--prepro_save_path', type=str, default='./prepro_vocab.json',
                        help='file path for saved preprocessor')
    return parser.parse_args()


if __name__ == '__main__':
    # Get arguments
    print('Getting arguments...')
    args = get_args()

    # make a dataset
    print('Importing dataset...')
    data = SentimentDataset(data=args.train_path)

    # preprocess and save word encodings
    preprocessor = Preprocessor(max_vocab=args.max_vocab)
    data = preprocessor.fit_transform(dataset=data)
    preprocessor.save(args.prepro_save_path)

    # validation split
    data.split_data(validation_count=args.validation_count)
    train_ds, val_ds = data.to_dataset()

    # to dataLoaders
    train_set = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_set = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print('Initializing model...')
    mod = SentimentModel(len(preprocessor.vocab2enc)+3, args.embedding_dim, args.hidden_dim)
    opt = Adam(mod.parameters(), lr=args.lr)

    print('Training...')
    fit(training=train_set, model=mod, validation=val_set, optimizer=opt, loss=torch.nn.BCELoss(), epochs=args.epochs)

    # Saving model
    print('Saving model...')
    torch.save(mod, args.model_save_path)

    print('Done!')
