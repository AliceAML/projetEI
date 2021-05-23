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
        f"French-Dictionary-master/dictionary/{cat}.txt",
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

eis = [l.strip() for l in open("liste_ei_ex").readlines()]

#%% recherche mot + similaire


def argmin(dico):
    # HYPERPARAM : si distance supérieur à 7, on renvoie rien
    return min(filter(lambda x: x[1] < 7, dico.items()), key=lambda x: x[1])[0]


def closest_match(ei):

    if ei.lower() in ["ielles", "iels"]:
        return "ils"
    elif ei.lower() in ["iel", "ielle"]:
        return "il"

    if not ei.isupper():  # si tout le mot n'est pas en majuscules
        base = (  # retire les E et S majuscules dans la fin du mot
            ei[0] + re.sub("[ES]", "", ei[1:])
        ).lower()  # met le tout en minucule
    else:
        base = ei.lower()

    if not base.isalpha():  # si y'a des séparateurs
        sep = [char for char in ei if not char.isalpha()][0]  # on récupère le sep
        base = base.split(sep)[0]  #  # on garde le premier segment de la base
        if ei.lower().endswith("s") and not base.endswith("s"):
            base += "s"  # on remet le pluriel si nécessaire
    prefix = base[:4]  # taille du préfixe -- HYPERPARAM A TESTER
    subdico = trie.values(prefix)
    scores = {w: edit_distance(base, w) for w in subdico}
    try:
        return argmin(scores)
    except:
        return base


#%%
def desinclufy_conll(conll, out):
    with open(conll, "r") as f:
        lines = f.readlines()

    with open(out, "w") as f:
        for line in lines:
            if line.startswith("#") or line == "\n":
                f.write(line)
                pass
            else:
                id, word, is_ei = line.split("\t")
                f.write(line.strip())
                if is_ei.strip() == "True":
                    f.write(f"\t{closest_match(word)}")
                f.write("\n")


# desinclufy_conll("corpus_ei_labelled.conll", "corpus_ei_deEI.conll")


# FIXME : COMMENT FAIRE POUR GARDER LE MEME FORMAT ? (maj, min...)


def deinclusify_text(conll, out):
    """Ajoute une ligne de texte désinclusifié pour chaque phrase

    Args:
        conll (filepath): Fichier déjà annoté, et avec les formes en EI
    """
    with open(conll, "r") as f, open(out, "w") as g:
        lines = [l.strip() for l in f.readlines()]
        sent = []
        for line in lines:
            if line.startswith("# doc"):
                print(line)
            if line.startswith("#"):
                sent = []
                g.write(line + "\n")
            elif line.startswith(tuple("0123456789")):
                g.write(line + "\n")
                num, word, is_EI, lemma = line.split("\t")
                sent.append(lemma)
            else:
                g.write(f"# text_no_ei =  {' '.join(sent)}\n")
                g.write(line + "\n")


deinclusify_text("1_corpus_annote_ei.conll", "2.corpus_no_ei.conll")
