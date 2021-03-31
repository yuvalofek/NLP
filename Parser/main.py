import argparse
import re
import logging


class Node:
    def __init__(self, content):
        self.right = None
        self.left = None
        self.content = content


class Parser:
    def __init__(self, grammar):
        self.rules = None
        self.table = None
        self.load_grammar(grammar)

    def load_grammar(self, grammar):
        self.rules = []
        with open(grammar) as f:
            logging.info('Grammar file opened')
            for line in f.readlines():
                ln = line.split(' --> ')
                right_arrow = re.sub('\n', '', ln[1])
                self.rules.append((ln[0], right_arrow.split(' ')))
        logging.debug(self.rules)
        logging.info('Grammar rules created')

    def parse(self, sentence):
        # sentence to list of words
        sentence_ = sentence.split(' ')
        # count of the number of words
        n = len(sentence_)

        # initialize table for parsing
        self.table = [[[] for i in range(n + 1)] for j in range(n + 1)]

        for j, word in enumerate(sentence_):
            for rule in self.rules:
                if word in rule[1]:
                    self.table[j-1][j].append(rule)
            for i in range(j-2, 0, -1):
                for k in range(i+1, j-1):
                    self.table[i][j].append()



def get_args():
    """
    Parse flags
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--grammar', type=str, default='./sampleGrammar.cnf', help='training file path')

    return parse.parse_args()


if __name__ == '__main__':
    logging.basicConfig(filename='./parser.log', level=logging.DEBUG)
    args = get_args()
    logging.info('Started')
    gl = Parser(args.grammar)
    logging.info('Grammar loaded')
    while True:
        sent = input('Enter a sentence: ')
        logging.info('Sentence input found')

        if sent == 'quit':
            logging.info('Quit detected')
            break
    print('Thank you! Have a nice day!')
    logging.info('Finished')
