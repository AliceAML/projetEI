"""
Determines if a word is an inclusive form or not.
"""
#%%
import re
import os

from load_dico import dico
from nltk.metrics.distance import edit_distance

#%%
def isEI(word: str) -> bool:
    """Returns true if word is an EI form.

    Args:
        wod (str): word to test
    """

    # terminaisons courantes non ambigües
    # à ignorer en fin de mot
    # pronoms, noms de domaine...
    # liste établie avec terminaisons_communes.py
    terminaisons = (
        "-être",
        "-même",
        "-mêmes",
        "-il",
        "-toi",
        "-moi",
        "-tu",
        "-elle",
        "-elles",
        "-ceux",
        "-celles",
        "-celle",
        "-uns",
        "-ils",
        "-lui",
        "-leur",
        "-on",
        "-je",
        "-vous",
        "-nous",
        "-midi",
        "-ci",
        "-ce",
        "-là",
        "-delà",
        ".com",
        ".org",
        ".net",
        ".be",
        ".fr",
        ".info",
        "-un",
        "-deux",
        "-trois",
        "-quatre",
        "-six",
        "-cinq",
        "-sept",
        "-huit",
        "-dix",
        "-neuf",
        "-onze",
        "-quatorze",
        "-douze",
        "treize",
        "quinze",
        "seize",
        "-bas",
        "-y",
        "-le",
        "-faire",
        "-vivre",
        "-ville",
        "-end",
        "-mère",
        "-père",
        "-gauche",
        "-droite",
        "-soi",
        "-fête",
        "-feu",
        "-fou",
        "-dire",
        "-dessus",
        "-bye",
    )

    # REGEX for EI forms with a separator (engagé-e-s, énervé·e)
    # word boundary +
    # (any letters separated by a separator) * possibly several times
    # word boundary
    ei_with_seps = r"\b([A-Za-zÀ-ÖØ-öø-ÿ]+[\-·\./\\][A-Za-zÀ-ÖØ-öø-ÿ]{0,6})+\b"

    ei_with_maj = r"\b[A-ZÀ-Ö]*[a-zø-ÿ]+E+[A-Za-zÀ-ÖØ-öø-ÿ]+\b"

    # EI capital letters : enervéEs
    if re.fullmatch(ei_with_maj, word):
        return True

    # EI with separators : énervé·es, énervé-e-s
    elif (
        not word.lower().endswith(terminaisons)
        and not isCompound(word)
        and re.fullmatch(ei_with_seps, word)
    ):
        first_part, *mid_part, last_part = re.split(r"[\-·\./\\]", word)
        if (
            # ignore proper noun
            not last_part.istitle()
            # attempt to ignore compound words
            # and len(last_part) < len(first_part)
            # checks that there's an "e" in the middle or last part
            and ("e" in last_part.lower() or "e" in "".join(mid_part).lower())
        ):
            return True

    # iels, ielles, iel... problème : parfois pas des réf générique..; mais rare
    elif (
        word.lower().startswith("iel") and word.lower().endswith("s") and len(word) <= 6
    ):
        return True
    elif word.lower() == "toustes":
        return True
    return False


# 3 TYPES A MATCHER
# - EI avec séparateurs
# - EI avec majuscules
# - EI "fusionnées" (sans marque typographique) > on verra plus tard
# utiliser un dictionnaire ? "toustes"
# iels ?


def isCompound(word: str) -> bool:
    if word in dico:
        return True
    if "-" in word:
        # sep = [char for char in word if not char.isalpha()][0]
        parts = word.split("-")
        if len(parts[-1]) > 3:
            return all((part.lower() in dico) for part in parts)
    return False


# # test isCompound
# with open("../log_ei_210507.txt") as f:
#     tests = [l.strip() for l in f.readlines()]

# nb_compounds = 0
# compounds = set()
# for w in tests:
#     res = isCompound(w)
#     if res:
#         nb_compounds += 1
#         compounds.add(w)

# print(*compounds, sep="\n")
# print(f"{nb_compounds=}")

# FIXME > liste de préfixe à enlever avant les tests ?

#%%
def dist_min(word: str) -> int:
    """Returns the minimum distance between a word and
    the closest word in the dictionary"""
    base = re.sub(r"\W", "", word).lower()
    subdico = dico.values(base[: len(base) // 2])
    scores = {w: edit_distance(base, w) for w in subdico}
    try:
        return min(scores.items(), key=lambda x: x[1])
    except:
        return (None, float("inf"))


# #%% tests dist_min
# with open("../log_uni.txt") as f:
#     tests = {l.strip() for l in f.readlines()}
# #%%
# dist_tests = []
# for w in tests:
#     dist_tests.append((w, dist_min(w)))
