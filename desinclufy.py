#%%
from pytrie import StringTrie
from nltk.metrics.distance import edit_distance
import pandas as pd
import string
import re

#%% import dictionnaire -> trie

mots_m = []

for cat in ("nouns", "pronouns", "adjectives", "ppas"):
    dico = pd.read_csv(
        f"../French-Dictionary-master/dictionary/{cat}.txt",
        sep=";",
        names=("mot", "genre", "nb"),
    )
    mots_m += list(
        dico[(dico["genre"] == "mas") | (dico["genre"] == "epi")]["mot"].values
    )


#%% création trie
trie = StringTrie()

for mot in mots_m:
    trie[mot] = mot

#%% import exemples

eis = [l.strip() for l in open("../liste_ei_ex").readlines()]

#%% recherche mot + similaire


def argmin(dico):
    try:
        return min(dico.items(), key=lambda x: x[1])[0]
    except:
        return None


def closest_match(ei):
    base = re.sub("[ES]", "", ei)  # retire les E et S majuscules
    if not base.isalpha():  # si y'a des séparateurs
        sep = [char for char in ei if not char.isalpha()][0]  # on récupère le sep
        base = base.split(sep)[0]  #  # on garde le premier segment de la base
        if ei.endswith("s") and not base.endswith("s"):
            base += "s"  # on remet le pluriel si nécessaire
    prefix = base[:4]  # taille du préfixe -- HYPERPARAM A TESTER
    subdico = trie.values(prefix)
    scores = {w: edit_distance(base, w) for w in subdico}
    return argmin(scores)


for ei in eis:
    print(ei, "->", closest_match(ei))