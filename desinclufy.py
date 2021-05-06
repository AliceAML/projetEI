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
    base = re.sub("[A-Z]", "", ei)
    if not base.isalpha():
        sep = [char for char in ei if not char.isalpha()][0]
        base = base.split(sep)[0]
        if ei.endswith("s") and not base.endswith("s"):
            base += "s"
    prefix = base[:4]
    subdico = trie.values(prefix)  # FIXME comment choisir ?
    scores = {w: edit_distance(base, w) for w in subdico}
    return argmin(scores)


for ei in eis:
    print(ei, "->", closest_match(ei))