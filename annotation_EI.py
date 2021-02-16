"""Lit un fichier texte et annote les formes EI

output : 1 fichier par fichier, avec liste de tuples (1 tuple par mot, avec annotation)
("Iels",True"),("sont",False),("arrivé·es",True),(".",False)
"""

import re

# TODO définir une regex surpuissante qui détecte les formes en EI

ei = re.compile(r"\b.*()|)

# TODO parcourir les fichiers

# parcourir les mots de chaque fichier
# TODO un peu de nettoyage (enlever caractères aberrants)

# TODO déterminer si chaque mot est en EI
# https://docs.python.org/3/howto/regex.html#match-versus-search    

# TODO write to file pour chaque fichier nom_du_fichier_annote

