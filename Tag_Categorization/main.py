import nltk
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import argparse

# from utils.DataReader import DataReader

from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('wordnet')
SEED = 31415
np.random.seed(SEED)

more_stop_words = ['could', "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", 'ought',
                   "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll",
                   "we're", "we've", "what's", "when's", "where's", "who's", "why's", 'would']


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


class CompoundTokenizer:
    """
    Perform tokenization
    """

    def __init__(self):
        # set classes for tokenization
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stemmer = nltk.stem.lancaster.LancasterStemmer()
        self.stop_words = list(nltk.corpus.stopwords.words('english'))
        self.stop_words.extend(more_stop_words)
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def __call__(self, document_str):
        # get the tokens
        tokens = self.tokenizer.tokenize(document_str.lower())
        # remove stop words
        filtered_words = [w for w in tokens if w not in self.stop_words]

        # stem
        stemmed = [self.stemmer.stem(w) for w in filtered_words]
        # lemmantized =  [self.lemmatizer.lemmatize(w) for w in filtered_words]

        # bi_gram
        # bi_gram = list(nltk.bigrams(lemmantized))
        # bi_gram = ([' '.join(tuple) for tuple in bi_gram])
        # lemmantized.extend(bi_gram)
        return stemmed


class CommonOperations:
    """
    Operations I used across classes and thought it would be useful to store together
    """

    @staticmethod
    def unique(list1):
        # intilize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list

    @staticmethod
    def normalize_dict(dict1):
        """
        normalize the values of the dictionary to have norm 1
        """
        norm = np.sqrt(np.sum(np.array(list(dict1.values())) ** 2))
        d = {key: value / norm for key, value in dict1.items()}
        return d

    @staticmethod
    def argmax(lst):
        # In terms of equality, we pick the first time the max was reached
        argument_max = -1
        max_v = -1
        for i, l in enumerate(lst):
            if l > max_v:
                argument_max = i
                max_v = l
        return argument_max


class VectorModel(CommonOperations):
    def __init__(self, comp_tokenizer, k):
        # token creating
        self.synonym_map = defaultdict(list)
        self.complex_tokenizer = comp_tokenizer
        self.idfs = None
        self.weights = None
        self.k = k
        self.drop_percentile = 3

    '''
    @staticmethod
    def normalize_dict(dict1):
        """
        normalize the values of the dictionary to have norm 1
        """
        norm = np.sqrt(np.sum(np.array(list(dict1.values())) ** 2))
        d = {key: value / norm for key, value in dict1.items()}
        return d
    
    @staticmethod
    def unique(list1):
        # intilize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        return unique_list
    '''

    def get_doc_tf(self, path, use_synonyms=False):
        """
        Reads in a document and returns its term frequencies
        :param path: document path
        :return: term frequencies for the document
        """
        # From a document path, get the documents tokens tf's
        # grab the document
        with open(path) as f:
            value = f.read()
        # tokenize
        tokens = self.complex_tokenizer(value)
        if use_synonyms:
            for idx, token in enumerate(tokens):
                tokens[idx] = self.synonym_map.get(token)
        # count the occurrences of each word
        word_counts = dict(Counter(tokens))
        num_tokens = len(tokens)
        # Get TF
        tf = {key: 1 + np.log(word_count / num_tokens) for key, word_count in word_counts.items()}
        return tf

    def set_weights(self, doc_paths):
        """
        Reads in all the documents in the input document path list, gets each document's term frequencies, then
        calculates input idfs and calculates each document's weight vector
        :param doc_paths: a list of document paths
        """
        # Number of documents
        n = len(doc_paths)

        # get TFs for all documents
        corp_tfs = [self.get_doc_tf(doc) for doc in doc_paths]

        # Get IDFs for all words seen
        # list of lists of the words seen in each document
        words = [list(d.keys()) for d in corp_tfs]
        # to one list (where the words repeat if in multiple docs)
        words = [item for sublist in words for item in sublist]
        words = Counter(words)
        self.idfs = {key: np.log(n / val) for key, val in words.items()}

        # calculate document TF*IDF weights
        self.weights = []
        for i, doc_TFs in enumerate(corp_tfs):
            # Getting the tf-idf
            doc_weights = {key: tf * self.idfs[key] ** self.k for key, tf in doc_TFs.items()}
            # normalizing per document & store in a list
            self.weights.append(self.normalize_dict(doc_weights))

        # drop lowest weights by percentile
        self.drop_low_weights(percentile=self.drop_percentile)

        # map synonyms of words in input vocabulary to the words in the vocabulary
        # self.map_synonyms()

    def map_synonyms(self):
        # create a dict of synonym to word mappings
        for word in self.idfs.keys():
            # get the synonyms
            synonyms = self.get_n_synonyms(word)
            # if we have synonyms
            if synonyms is not None or not synonyms:
                # append to list in case multiple words have the same synonyms
                for synonym in synonyms:
                    self.synonym_map[synonym].append(word)

        for word in self.idfs.keys():
            # words in the input vocabulary should always map to themselves
            self.synonym_map[word] = [word]

        # in case multiple words in our test docs have the same synonyms, set the synonyms to map to the words
        # with the lowest idf --> least influence our results
        for synonym, words in self.synonym_map.items():
            # if multiple words in the input vocab map to the same synonym
            if len(words) > 1:
                word_weights = []
                for word in words:
                    word_weights.append(self.idfs[word])
                self.synonym_map[synonym] = words[self.argmax(word_weights)]
            # if the synonyms map to one word
            elif len(words) == 1:
                self.synonym_map[synonym] = words[0]

    def get_inverse_weights(self, corp_tfs):
        vocab = {}
        for word in self.idfs.keys():
            vocab[word] = []
            for index, d in enumerate(corp_tfs):
                if word in d.keys():
                    weight = self.weights[index][word]
                    vocab[word].append({'id': index, 'w': weight})
        return vocab

    def get_n_synonyms(self, word, top_n=-1):
        # Then, we're going to use the term "program" to find synsets like so:
        synonyms = wordnet.synsets(word)
        synonyms = [syn.lemmas()[0].name() for syn in synonyms]
        return self.unique(synonyms)[:top_n]

    def drop_low_weights(self, percentile=3):
        weights = [list(doc_weight.values()) for doc_weight in self.weights]
        weights = np.array([item for sublist in weights for item in sublist])
        min_weight = np.percentile(weights, percentile, interpolation='midpoint')
        for idx, document_weight in enumerate(self.weights):
            self.weights[idx] = {word: weight for word, weight in self.weights[idx].items() if weight > min_weight}

    def get_test_weights(self, doc_paths):
        """
        Reads in the documents specified in the document path and calculates their weight vector based on the stored
        idf values.
        :param doc_paths: document path list
        :return: a list of test document weights
        """
        # check that we have the trained idfs
        assert (self.idfs is not None)
        tfs = [self.get_doc_tf(doc, use_synonyms=False) for doc in doc_paths]

        test_weights = []
        for doc_TFs in tfs:
            # Getting the tf-idf
            doc_weights = {}
            for word, tf in doc_TFs.items():
                doc_weights[word] = tf * self.idfs.get(word, 0.0)
            # normalizing per document & store in a list
            test_weights.append(self.normalize_dict(doc_weights))
        return test_weights

    def save_vm(self, out_path='./'):
        """
        Saves the vector model training weights and idfs
        :param out_path: output path for the saved file
        :return: 
        """
        packet = {'w': self.weights, 'idfs': self.idfs}
        with open(out_path + 'weights.json', 'w') as fout:
            json.dump(packet, fout)

    def load_vm(self, in_path='./'):
        """
        Loads a vector model from a save file
        :param in_path: input save file
        :return:
        """
        # load the weights from a .json file
        with open(in_path + 'weights.json', 'r') as fin:
            packet = json.loads(fin.read())
        self.weights = packet['w']
        self.idfs = packet['idfs']


class Rocchio(CommonOperations):
    def __init__(self, k):
        self.compound_tokenizer = CompoundTokenizer()
        self.category_weights = {}
        self.available_labels = None
        self.vm = VectorModel(self.compound_tokenizer, k)

    @staticmethod
    def dict_add(dict1, dict2):
        # Adds the keys of two dictionaries. If a key doesn't exist in one dict,
        # the output dictionary's key will have the value that the key held in the
        # other dict (the one where the key did exist in). Else, the sum of the
        # two values is set to output key value:
        # Cases: key in dict1, key in dict2, or key in both.
        out_dict = {}
        for key in dict2.keys():
            if key in dict1.keys():
                out_dict[key] = dict2[key] + dict1[key]
            else:
                out_dict[key] = dict2[key]
        for key in dict1.keys():
            if key not in dict2.keys():
                out_dict[key] = dict1[key]
        return out_dict

    def train(self, doc_paths, labels):
        # doc_paths, labels = dataset.get_train()
        self.available_labels = self.unique(labels)

        # get the indices that match each label
        label_indices = {label: [i for i, x in enumerate(labels) if x == label]
                         for label in self.available_labels}

        # Generate the weights for each of the tokens in the training set
        self.vm.set_weights(doc_paths)
        # add the weights of the documents to get category weights & normalize
        for label, indices in label_indices.items():
            self.category_weights[label] = {}
            for i in indices:
                self.category_weights[label] = self.dict_add(self.category_weights[label],
                                                             self.vm.weights[i])
            # normalize
            self.category_weights[label] = self.normalize_dict(self.category_weights[label] )

    @staticmethod
    def dict_cos_sim(dict1, dict2):
        dict_smaller = dict1 if len(dict1) < len(dict2) else dict2
        dict_bigger = dict2 if len(dict1) < len(dict2) else dict1
        # since weight vectors are normalized, we only need to take the inner
        # product, which is the sum of products of the vector elements
        similarity = 0
        for key, value in dict_smaller.items():
            # get key in bigger dict if exists, else zero and multiply by val in
            # smaller dict
            similarity += (dict_bigger.get(key, 0.0) * value) ** 0.5
        return similarity

    def test(self, doc_paths, write_out=False, write_path='./output.labels'):
        # calculate the document weights
        vm_weights = self.vm.get_test_weights(doc_paths)
        predictions = []
        # loop over the test documents
        for doc_w in vm_weights:
            # for each document calculate the
            cat_similarities = []
            for label in self.available_labels:
                cat_similarities.append(self.dict_cos_sim(self.category_weights[label], doc_w))
            predictions.append(self.available_labels[self.argmax(cat_similarities)])
        if write_out:
            self.write_out(doc_paths, predictions, write_path)
        return predictions

    @staticmethod
    def write_out(doc_paths, predictions, write_path):
        """
        Writes out the document pats and predictions in the format prescribed
        """
        df = pd.DataFrame({'path': doc_paths, 'labels': predictions})
        df.to_csv(write_path, sep=' ', header=False, index=False)

    def save_rocchio(self, out_path='./'):
        # save the weights to a .json
        self.vm.save_vm()
        packet = {'w': self.category_weights,
                  'labels': self.available_labels}
        with open(out_path + 'Rocchio.json', 'w') as fout:
            json.dump(packet, fout)

    def load_rocchio(self, in_path='./'):
        # load the weights from a .json file
        with open(in_path + 'Rocchio.json', 'r') as fin:
            packet = json.loads(fin.read())
        self.category_weights = packet['w']
        self.available_labels = packet['labels']
        self.vm.load_vm()


class ParameterTuner:
    def __init__(self, k_fold=6, model=Rocchio):
        self.k_fold = k_fold
        self.skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=SEED)
        self.model = model
        self.accuracies = None
        self.best_param = None
        # Move from cross-validation to full dataset we might want to change the
        # exponent
        self.correction = 0

    def tune(self, x, labels, param):
        self.accuracies = np.zeros((self.k_fold, len(param)))

        # Stratified K-fold cross validation
        for k_f, (train_idx, test_idx) in tqdm(enumerate(self.skf.split(x, labels)),
                                               total=self.k_fold, position=0):
            for p_idx, p in enumerate(param):
                x_tr = [x[i] for i in train_idx]
                x_te = [x[i] for i in test_idx]
                y_tr = [labels[i] for i in train_idx]
                y_te = [labels[i] for i in test_idx]

                r1 = self.model(p)
                r1.train(x_tr, y_tr)
                self.accuracies[k_f, p_idx] = self.evaluate(r1, x_te, y_te)

        self.best_param = param[self.accuracies.mean(axis=0).argmax()] + self.correction
        # print('Max accuracy from cross-validation: ', self.accuracies.max())
        # print('Selected idf exponent of: {}'.format(self.best_param))

        best_model = self.model(self.best_param)
        best_model.train(x, labels)
        return best_model

    @staticmethod
    def evaluate(model, x, labels):
        predictions = model.test(x)
        correct = np.array([prediction == labels[i] for i, prediction in enumerate(predictions)])
        return correct.mean()


def get_args():
    """
    Parse flags
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='./corpus1_train.labels')
    parser.add_argument('--output_path', type=str, default='./output.labels')
    parser.add_argument('--test_path', type=str, default='./corpus1_test.list')

    return parser.parse_args()


if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # read in the training labels file
    training = DataReader(args.input_path)
    print('Training data obtained from: {}'.format(args.input_path))

    # hyper-parameter tuning - idf exponent in TF-IDF weights (weight = tf*(idf**k))
    N_exponents = 2
    k_fold = 2

    Ks = 0.1 + np.linspace(0, 0.7, N_exponents)
    # tune the exponent and train the model on best value
    print('Tuning parameters and training model...')
    tuner = ParameterTuner(k_fold=k_fold)
    X, y = training.get_data()
    r = tuner.tune(X, y, Ks)

    # test the model on the
    print('Predicting labels for test data from: {}'.format(args.test_path))
    testing = DataReader(args.test_path)
    X_te, _ = testing.get_data()
    r.test(X_te, write_out=True, write_path=args.output_path)
    print('Output saved to: {}'.format(args.output_path))
