# deep learning sentiment analysis project - Yuval Ofek

# read in data
import pandas as pd

# debugging and running code
import logging
import argparse

# preprocessing
import re
import string
import nltk
from sklearn.model_selection import train_test_split

# Set up stop words
nltk.download('stopwords', quiet=True)
more_stop_words = ['could', "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", 'ought',
                   "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll",
                   "we're", "we've", "what's", "when's", "where's", "who's", "why's", 'would']
stop_words = list(nltk.corpus.stopwords.words('english'))
stop_words.extend(more_stop_words)

"""
dataset: https://www.kaggle.com/kazanova/sentiment140
1.6 Million tweets

It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
* ids: The id of the tweet ( 2087)
* date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
* flag: The query (lyx). If there is no query, then this value is NO_QUERY.
* user: the user that tweeted (robotickilldozr)
* text: the text of the tweet (Lyx is cool) [av
"""


class Dataset:
    def __init__(self, dataset_size=None, test_split=0.2):
        # Url regex courtesy of https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to
        # -match-a-url
        self.urlPattern = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}" \
                          r"\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        self.userPattern = r'@[\S]+'

        # preprocessing
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+',)
        self.stop_words = stop_words
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

        # init parameters
        self.dataset_size = dataset_size
        self.test_split = test_split

        # unset
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        logging.info('Dataset object initialized')

    def load_data(self, file_path):
        # bring in the data
        data = pd.read_csv(file_path,
                           encoding='latin',
                           names=['polarity', 'id', 'date', 'query', 'user', 'text'])
        # save only the polarity and text
        data = data[['polarity', 'text']]

        # drop neutral columns
        data.drop(data[data['polarity'] == 2].index, inplace=True)

        # Remap polarity to 0-1
        data['polarity'] = data['polarity'].replace(4, 1)

        # shuffle
        self.data = data.sample(frac=1).reset_index(drop=True)

        # crop length
        if self.dataset_size is not None:
            self.data = self.data[:min(self.dataset_size, len(self.data))]
        logging.info('data loaded')
        return self.data

    def preprocess_tweet(self, text):
        # lowercase
        text = text.lower()
        # remove urls and usernames
        text = re.sub(self.urlPattern, ' ', text)
        text = re.sub(self.userPattern, ' ', text)
        # remove punctuation
        text = text.translate(string.punctuation)
        # tokenize, remove stop words, & stem
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        logging.debug('{} preprocessed'.format(text))
        return tokens

    def preprocess_dataset(self):
        self.data['text'] = self.data['text'].apply(lambda x: self.preprocess_tweet(x))
        # conveniently this also pads the text
        logging.info('dataset preprocessed')
        return

    def split_dataset(self):
        test_size = int(self.dataset_size*self.test_split)
        x = self.data['text'].tolist()
        y = self.data['polarity'].tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x,
                                                                                y,
                                                                                test_size=test_size,
                                                                                stratify=y)
        self.data = None
        logging.info('dataset split')

    def save_dataset(self, train_path='./train.csv', test_path='./test.csv'):
        train = pd.DataFrame(self.X_train, self.y_train)
        test = pd.DataFrame(self.X_test, self.y_test)
        train.to_csv(train_path)
        test.to_csv(test_path)
        logging.info('training and testing datasets saved')


def get_args():
    """
    Parse flags
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, default='./sentiment_data.csv', help='training file path')
    parse.add_argument('--size', type=int, default=500_000, help='total number of tweets in final dataset')
    parse.add_argument('--test_split', type=float, default=0.2, help='fraction of dataset to save for test set')
    parsed = parse.parse_args()
    logging.info('parsed arguments')
    return parsed


if __name__ == '__main__':
    logging.basicConfig(filename='./sentiment_analysis.log', level=logging.INFO, filemode='w')
    logging.info('Started')
    args = get_args()

    print('Initializing...')
    dataset = Dataset(dataset_size=args.size, test_split=args.test_split)
    print('Loading data...')
    dataset.load_data(file_path=args.data_path)
    print('preprocessing...')
    dataset.preprocess_dataset()
    print('Making train and test sets...')
    dataset.split_dataset()
    print('Saving...')
    dataset.save_dataset()
    print('Done!')
    logging.info('Finished')
