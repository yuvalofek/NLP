import argparse
import re
import logging


class GrammarLoader:
    def __init__(self, grammar_path):
        self.rules = []
        with open(grammar_path) as f:
            for line in f.readlines():
                ln = line.split(' --> ')
                right_arrow = re.sub('\n', '', ln[1])
                self.rules.append((ln[0], right_arrow.split(' ')))

        logging.debug(self.rules)
        logging.info('Grammar Rules Created')

class Parser:
    def __init__(self, grammar):
        self.grammar = grammar

    def parse(self, sentence):
        sentence_list = sentence.split(' ')
        for word in sentence_list:
            None


def get_args():
    """
    Parse flags
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--grammar', type=str, default='./sampleGrammer.cnf', help='training file path')

    return parse.parse_args()


if __name__ == '__main__':
    logging.basicConfig(filename='./parser.log', level=logging.DEBUG)
    args = get_args()
    logging.info('Started')
    gl = GrammarLoader(args.grammar)
    logging.info('Grammar loaded')
    while True:
        sent = input('Enter a sentence: ')
        logging.info('Sentence input found')

        if sent == 'quit':
            logging.info('Quit detected')
            break
    print('Thank you! Have a nice day!')
    logging.info('Finished')
