"""
Determines if a word is an inclusive form or not.
"""

import re
import os


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
    elif not word.lower().endswith(terminaisons) and re.fullmatch(ei_with_seps, word):
        first_part, *mid_part, last_part = re.split(r"[\-·\./\\]", word)
        if (
            # ignore proper noun
            not last_part.istitle()
            # attempt to ignore compound words
            # and len(last_part) < len(first_part)
            # checks that there's an "e" in the middle or last part
            and ("e" in last_part or "e" in "".join(mid_part).lower())
        ):
            return True

    # iel, iels, ielle, ielles (?)
    elif word.lower().startswith("iel") and len(word) <= 6:
        return True

    return False


# 3 TYPES A MATCHER
# - EI avec séparateurs
# - EI avec majuscules
# - EI "fusionnées" (sans marque typographique) > on verra plus tard
# utiliser un dictionnaire ? "toustes"
# iels ?
