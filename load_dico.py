"""Script that loads a trie of French words,
to be used in other scripts"""

import os
import logging
from pytrie import StringTrie

path = "../French-Dictionary-master/dictionary"

files = os.listdir(path)

dico = StringTrie()

for file in files:
    logging.info(f"Loading {file}")
    with open(path + "/" + file) as f:
        mots = [line.split(";")[0].strip() for line in f.readlines()]
        for mot in mots:
            dico[mot] = mot

logging.info("Dico loaded")
