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
        sentence_ = sentence.split(' ')
        n = len(sentence_)

        # initialize?


        for word_i in range(1, n):
            None


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
