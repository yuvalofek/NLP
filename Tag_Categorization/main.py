from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import argparse

import nltk
from nltk.corpus import wordnet

# from utils.DataReader import DataReader

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

    def __call__(self, document_str):
        """
        Tokenize and input string
        :param document_str: (str) input string
        :return: (list of str) list of tokens
        """

        # get the tokens
        tokens = self.tokenizer.tokenize(document_str.lower())
        # remove stop words
        filtered_words = [w for w in tokens if w not in self.stop_words]
        # stem
        stemmed = [self.stemmer.stem(w) for w in filtered_words]
        return stemmed


class CommonOperations:
    """
    Operations I used across classes and thought it would be useful to store together
    """

    @staticmethod
    def unique(list1):
        """
        :param list1: (list) arbitrary list
        :return: list of all the unique elements in the input list
        """
        # initialize a null list
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
        Normalize the values of the dictionary to have norm 1
        :param: (dict) dictionary to normalize
        :return: (dict) normalized dictionary (L-2 Norm)
        """
        norm = np.sqrt(np.sum(np.array(list(dict1.values())) ** 2))
        d = {key: value / norm for key, value in dict1.items()}
        return d

    @staticmethod
    def argmax(lst):
        """
        List argmax - returns the index of the largest element in the list
        """
        # In terms of equality, we pick the first time the max was reached
        argument_max = -1
        max_v = -1
        for i, l in enumerate(lst):
            if l > max_v:
                argument_max = i
                max_v = l
        return argument_max


class VectorModel(CommonOperations):
    def __init__(self, comp_tokenizer, k=0.2):
        # vector model parameters
        self.complex_tokenizer = comp_tokenizer
        self.k = k
        self.drop_percentile = 3

        # storage
        self.idfs = None
        self.weights = None

        # synonym mapping
        self.synonym_map = defaultdict(list)
        self.substitute_synonyms = True

    def get_doc_tf(self, path, sub_synonyms=False):
        """
        Reads in a document and returns its term frequencies
        :param path: (str) document path
        :param sub_synonyms: (bool) substitute tokens with synonyms in self.synonym_map
        :return: term frequencies for the document
        """
        # From a document path, get the documents tokens tf's
        # grab the document
        with open(path) as f:
            value = f.read()
        # tokenize
        tokens = self.complex_tokenizer(value)

        # if we want to substitute input words for their synonyms in the training synonym mapping
        if sub_synonyms:
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
        :param doc_paths: (list of str) a list of document paths
        """
        # Number of documents
        n = len(doc_paths)

        # get TFs for all documents
        corp_tfs = [self.get_doc_tf(doc) for doc in doc_paths]

        # Get IDFs for all words seen
        # list of lists of the words seen in each document
        words = []
        for document in corp_tfs:
            words.extend(list(document.keys()))

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
        """
        Use the words in the self.idfs to create an inverted dictionary from synonyms to self.idfs words
        """
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
        """
        Reads through the weights and inverts them so each word is a dict of the document index and the corresponding
        word weight in the document.
        :param corp_tfs: (list of dicts) corpora tfs list
        """
        vocab = {}
        for word in self.idfs.keys():
            vocab[word] = []
            for index, d in enumerate(corp_tfs):
                if word in d.keys():
                    weight = self.weights[index][word]
                    vocab[word].append({'id': index, 'w': weight})
        return vocab

    def get_n_synonyms(self, word, top_n=-1):
        """
        Returns the top N synonyms for an input word
        :param word: (str) input word
        :param top_n: (int) top n synonyms to return at best scenario (will return fewer if word doesn't have synonyms)
        """
        # Then, we're going to use the term "program" to find synsets like so:
        synonyms = wordnet.synsets(word)
        synonyms = [syn.lemmas()[0].name() for syn in synonyms]
        un_synonyms = self.unique(synonyms)
        # number of elements to return
        num_elements = min(len(un_synonyms), top_n)
        return un_synonyms[:num_elements]

    def drop_low_weights(self, percentile=3):
        """
        Drops the lowest weights of self.weights by percentile
        :param percentile: (float) percentile to drop
        """
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
        tfs = [self.get_doc_tf(doc, sub_synonyms=False) for doc in doc_paths]

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
        self.vm = VectorModel(self.compound_tokenizer, k)

        self.category_weights = {}
        self.available_labels = None

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
            self.category_weights[label] = self.normalize_dict(self.category_weights[label])

    def test(self, doc_paths, write_out=False, write_path='./output.labels'):
        # calculate the document weights
        vm_weights = self.vm.get_test_weights(doc_paths)
        predictions = []
        # loop over the test documents
        for doc_w in tqdm(vm_weights, position=0):
            # for each document calculate the
            cat_similarities = []
            for label in self.available_labels:
                cat_similarities.append(self.dict_cos_sim(self.category_weights[label], doc_w))
            predictions.append(self.available_labels[self.argmax(cat_similarities)])
        if write_out:
            self.write_out(doc_paths, predictions, write_path)
        return predictions

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

    def tune(self, x, labels, param):
        """
        Tunes a single model parameter based on an input parameter list and stratified cross validation
        :param x: (list of str) list of document paths
        :param labels: (list of str) list of document labels
        :param param: (iterable) parameter values to test model on
        :return: (model) trained model trained with the best-parameter found through stratified cross validation
        """
        self.accuracies = np.zeros((self.k_fold, len(param)))

        # Stratified K-fold cross validation
        for k_f, (train_idx, test_idx) in tqdm(enumerate(self.skf.split(x, labels)),
                                               total=self.k_fold, position=0):
            for p_idx, p in enumerate(param):
                # set the training and validation data & labels
                x_tr = [x[i] for i in train_idx]
                x_te = [x[i] for i in test_idx]
                y_tr = [labels[i] for i in train_idx]
                y_te = [labels[i] for i in test_idx]

                # initialize and train model
                r1 = self.model(p)
                r1.train(x_tr, y_tr)
                # add accuracy to array
                self.accuracies[k_f, p_idx] = self.evaluate(r1, x_te, y_te)

        # Save the best parameter
        self.best_param = param[self.accuracies.mean(axis=0).argmax()]

        # return a trained model on the best parameter - trained on all the data
        best_model = self.model(self.best_param)
        best_model.train(x, labels)
        return best_model

    @staticmethod
    def evaluate(model, x, labels):
        """
        Evaluate a model's accuracy on a test set
        :param model: (model) model to evaluate
        :param x: (list of str) list of test data
        :param labels: (list of str) list of test labels
        """
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
    X, y = training.get_data()

    # hyper-parameter tuning - idf exponent in TF-IDF weights (weight = tf*(idf**k))
    N_exponents = 2
    K_fold = 2

    Ks = 0.1 + np.linspace(0, 0.7, N_exponents)
    # tune the exponent and train the model on best value
    # print('Tuning parameters and training model...')
    # tuner = ParameterTuner(k_fold=K_fold)
    # r = tuner.tune(X, y, Ks)

    print('Training model...')
    r = Rocchio(0.2)
    r.train(X, y)

    # test the model on the
    print('Predicting labels for test data from: {}'.format(args.test_path))
    testing = DataReader(args.test_path)
    X_te, _ = testing.get_data()
    r.test(X_te, write_out=True, write_path=args.output_path)
    print('Output saved to: {}'.format(args.output_path))
