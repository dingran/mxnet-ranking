from random import choice
from os import path


def words_from(resource_name):
    full_path = path.join(path.dirname(__file__), resource_name)
    with open(full_path, 'r') as f:
        return [l.rstrip('\n') for l in f.readlines() if l and not l.startswith('#')]


ADJECTIVES = words_from('DatasetId.adjectives.txt')
NOUNS = words_from('DatasetId.nouns.txt')


def random_phrase():
    return choice(ADJECTIVES) + "-" + choice(NOUNS)
