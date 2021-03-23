"""
Création d'une liste de mots pas dans le dictionnaire
à modifier manuellement pour faire une liste de formes en EI

INUTILE
"""

import os
import re

#%% import des dicos
dico_files = os.listdir("French-Dictionary-master/dictionary")
dico_liste = []

for file in dico_files:
    with open("French-Dictionary-master/dictionary/" + file) as f:
        lines = f.readlines()
        for line in lines:
            mot = re.search(
                r"([^;\n]+)", line
            )  # récupère le mot (avant le point virgule)
            dico_liste.append(mot.group(1))


def no_title(string):
    """makes first letter of a word lowercase"""
    return string[0].lower() + string[1:]


#%% création liste de mots pas dans le dico

corpus = os.listdir("corpus/infokiosque")
words = []

for text in corpus:
    with open("corpus/infokiosque/" + text) as f:
        content = f.read()
        words += re.split(r"(\s+|'|’)+", content)

for word in words:
    stripped_word = word.strip(";.,?!\"'«»() ")
    if len(stripped_word) > 1 and stripped_word.isalpha():
        clean_word = no_title(stripped_word)
        if clean_word not in dico_liste and word not in dico_liste:
            print(clean_word)


# %%
