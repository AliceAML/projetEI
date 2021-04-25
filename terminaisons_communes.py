from collections import defaultdict
import re

"""
Programme qui affiche les terminaisons les plus courantes
d'une liste de mots composés.
"""


with open("liste_annotee5.txt") as f:
    words = f.readlines()

endings = defaultdict(int)

for word in words:
    word = word.strip()
    split_word = re.split(r"[\-·\./\\]", word)
    if len(split_word) > 1:
        *_, last_part = split_word
    endings[last_part.lower()] += 1

print({k: v for (k, v) in sorted(endings.items(), key=lambda x: -x[1]) if v > 1})
