import argparse
import logging
from collections import defaultdict
import time


class Node:
    def __init__(self, root, right, left, terminal):
        self.root = root
        self.right = right
        self.left = left
        self.terminal = terminal


class Parser:
    def __init__(self, grammar):
        self.table = None
        self.grammar = self.load_grammar(grammar)

    @staticmethod
    def load_grammar(grammar_file):
        grammar = defaultdict(list)
        with open(grammar_file) as f:
            logging.info('Grammar file opened')
            for line in f.readlines():
                # if there are comments in the grammar file, skip them
                if line[0] == '#':
                    logging.info('Grammar file comment skipped')
                    continue
                # for content lines:
                rule = line.split(' --> ')
                # strip any un-needed characters
                left_side = rule[0].strip()
                right_side = rule[1].strip()
                # split the right side on space to make into a list
                right_side = right_side.split(' ')
                # add to grammar dictionary
                grammar[left_side].append(right_side)
        logging.debug(grammar)
        logging.info('Grammar rules created')
        return grammar

    def parse(self, sentence):
        # sentence to list of words
        sentence_ = sentence.split(' ')
        logging.info('Sentence split into: {}'.format(sentence_))
        # count of the number of words
        n = len(sentence_)

        # initialize table for parsing
        table = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
        table2 = [[[] for _ in range(n + 1)] for _ in range(n + 1)]

        for j in range(1, n+1):
            logging.debug('j = {j}'.format(j=j))
            word = sentence_[j-1]
            # {A| A ---> word in grammar}
            for left_side, right_side in self.grammar.items():
                logging.debug('word: {}, rule: {}'.format(word, right_side))
                if [word] in right_side:
                    logging.info('Word in grammar rules')
                    table[j-1][j].append(left_side)
                    table2[j-1][j].append(Node(left_side, None, None, word))

            for i in range(j - 2, -1, -1):  # j-2 to 0
                logging.debug('i={i}'.format(i=i))
                for k in range(i + 1, j):
                    logging.debug('k={k}'.format(k=k))
                    # {A| A --> BC in grammar & B in table[i,k] & C in table[k,j]}
                    for left_side, right_side in self.grammar.items():
                        for right_element in right_side:
                            # if we have 2 elements in the rule right side
                            if len(right_element) == 2:
                                B = right_element[0]
                                C = right_element[1]
                                logging.debug('B: {}, C: {}'.format(B, C))
                                logging.debug('table[i][k]: {}, table[k][j]: {}'.format(table[i][k], table[k][j]))
                                if B in table[i][k] and C in table[k][j]:
                                    table[i][j].append(left_side)
                                    logging.info('Rule added to table[{i}][{j}]:{rule}'.format(i=i, j=j, rule=left_side))
                                    for b in table2[i][k]:
                                        for c in table2[k][j]:
                                            if b.root == B and c.root == C:
                                                table2[i][j].append(Node(left_side, b, c, None))
                                                logging.info('Node added to table2[{i}][{j}]:{rule}'.format(i=i, j=j, rule=left_side))
        logging.debug('Parse tree: {tree}'.format(tree=table))
        logging.debug('Parse tree2: {tree}'.format(tree=table2))
        self.print_parse_trees(table2[0][n], 1)

    def get_parse_tree(self, root, indent):
        if root.status:
            return '(' + root.root + ' ' + root.terminal + ')'

        # Calculates the new indent factors that we need to pass forward.
        new1 = indent + 2 + len(root.left.root)  # len(tree[1][0])
        new2 = indent + 2 + len(root.right.root)  # len(tree[2][0])
        left = self.get_parse_tree(root.left, new1)
        right = self.get_parse_tree(root.right, new2)
        return '(' + root.root + ' ' + left + '\n' \
               + ' ' * indent + right + ')'

    def print_parse_trees(self, nodes_back, indent):
        check = False
        for node in nodes_back:
            if node.root == 'S':
                print(self.get_parse_tree(node, indent*3))
                print()
                check = True

        if not check:
            print('NO VALID PARSES')


def get_args():
    """
    Parse flags
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--grammar', type=str, default='./sampleGrammar.cnf', help='training file path')

    return parse.parse_args()


if __name__ == '__main__':
    # configure logging
    logging.basicConfig(filename='./parser.log', level=logging.DEBUG, filemode='w')
    logging.info('Started')

    # get grammar file path
    args = get_args()

    # initialize parser with the grammar
    parser = Parser(args.grammar)
    logging.info('Grammar loaded')

    # prompt for how to print parse trees
    parse_tree_config = input('Do you want textual parse trees to be displayed (y/n)?: ')
    parse_tree_config = parse_tree_config == 'y'
    logging.info('Parse trees displayed: '+str(parse_tree_config))

    # work loop
    while True:
        # prompt user for sentence
        sent = input('Enter a sentence: ')
        logging.info('Sentence input found: {sent}'.format(sent=sent))

        if sent == 'quit':
            logging.info('Quit detected')
            break

        parser.parse(sent)
        logging.info('Sentence Parsed')

    print('Thank you! Have a nice day!')
    logging.info('Finished')
