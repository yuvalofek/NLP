import pandas as pd
import argparse
import logging
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import string

# Set up stop words
nltk.download('stopwords', quiet=True)
more_stop_words = ['could', "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", 'ought',
                   "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll",
                   "we're", "we've", "what's", "when's", "where's", "who's", "why's", 'would']
stop_words = list(nltk.corpus.stopwords.words('english'))
stop_words.extend(more_stop_words)

"""
dataset: https://www.kaggle.com/kazanova/sentiment140

It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
* ids: The id of the tweet ( 2087)
* date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
* flag: The query (lyx). If there is no query, then this value is NO_QUERY.
* user: the user that tweeted (robotickilldozr)
* text: the text of the tweet (Lyx is cool) [av
"""


class Dataset():
    def __init__(self, N):
        # Url regex courtesy of https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to
        # -match-a-url
        self.urlPattern = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        self.userPattern = '@[\S]+'

        # preprocessing
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stop_words = stop_words
        self.stemmer = nltk.stem.lancaster.LancasterStemmer()

        # init parameters
        self.N = N

        # unset
        self.data = None
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
        data = data.sample(frac=1).reset_index(drop=True)

        # crop length
        self.data = data[:self.N]
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
        tokens = [w for w in tokens if w not in self.stop_words]
        tokens = [self.stemmer.stem(w) for w in tokens]
        logging.debug('{} preprocessed'.format(text))
        return ' '.join(tokens)

    def preprocess_dataset(self):
        self.data['text'] = self.data['text'].apply(lambda x: self.preprocess_tweet(x))
        logging.info('dataset preprocessed')


def check_word_count(text):
    return len(text.split(' '))


def get_args():
    """
    Parse flags
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', type=str, default='./sentiment_data.csv', help='training file path')

    return parse.parse_args()


if __name__ == '__main__':
    # length of input dataset
    N = 10000
    args = get_args()

    dataset = Dataset(N)
    dataset.load_data(args.data)

    dataset.preprocess_dataset()
    print(dataset.data)

    word_count = dataset.data['text'].apply(check_word_count)

    plt.figure()
    plt.hist(word_count)
    plt.show()
    # we see that the max word count is under 30 words
