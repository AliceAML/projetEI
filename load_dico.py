"""Script that loads a set of French words,
to be used in other scripts"""

import os
import logging

path = "../French-Dictionary-master/dictionary"

files = os.listdir(path)

dico = set()

for file in files:
    logging.info(f"Loading {file}")
    with open(path + "/" + file) as f:
        mots = [line.split(";")[0].strip() for line in f.readlines()]
        dico.update(mots)

logging.info("Dico loaded")