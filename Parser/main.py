import argparse


class Parser:
    def __init__(self, grammar):
        self.grammar = grammar


def get_args():
    """
    Parse flags
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--grammar_path', type=str, default=None, help='training file path')

    return parse.parse_args()


if __name__ == '__main__':
    args = get_args()

    while True:
        sentence = input('Enter a sentence: ')

        if sentence == 'quit':
            break
