"""
Annotates texts to indicate which words are EI forms.
"""

import sys

sys.path.append("../codeEI")

import regex as re
from findEI import isEI


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as data:
        text = data.read()
        clean = text.replace("\t", " ").replace("\n", " ")
        # output = clean.lower()
        output = clean

    return output


# if script launched as main
if __name__ == "__main__":
    name = sys.argv[1]

text = load_file(name)
sentences = re.split(r"(?<=\.|\?|\!)\s", text)


# print("text 100", text[0:100])
# à propos de la regex : \b pour identifier les fins de mots dans une string (utilisable sur une pharse, il faut qu'il y ai
# un séparateur. si on passe sur des items : pas de sep)
# print("sentences 10", *sentences[:10], sep="\n\n")


def tokenize(sentence: str) -> list:
    """
    tokenize and keep ponctuation marks, but not whistespaces

    Args:
        sentence (str): sentence to tokenize

    Returns:
        list: tokenized sentence, with ponctuation
    """
    return [
        segment
        for segment in re.split(r"([^\w\-·\.\/\\])", sentence)
        if segment.strip()
    ]


def labelling(sentences: list) -> list:
    """
    Label words in a list of sentences (EI or not EI)
    Returns a list of annotated sentences.
    """
    out = []
    for one_sentence in sentences:
        tempwords = tokenize(one_sentence)
        tempgold = []
        for index, word in enumerate(tempwords):
            if isEI(word):
                tempgold.append(index)
        tempdict = dict()
        tempdict["words"] = tempwords
        tempdict["gold"] = tempgold
        out.append(tempdict)
    return out


labels = labelling(sentences)

if sys.argv[2] == "-v":
    print(labels)


for i in labels:
    if i["gold"]:
        for j in i["gold"]:
            print(i["words"][j])


# output = list(map(lambda x: x.replace('\t','').replace('\n',''),text))

# 1  nettoyer le texte (tabulation? passages de lignes?)
#       phrases coupees par des sauts de lignes = ok
#       phrases de titres : elles n'ont pas de points, ça les fusionne
# 2  mettre en forme de corpus (par textes entiers?)
# 3  estimer les meilleures regex

# \n([a-zA-Z0-9]+|[A-Z]\+[a-zA-Z0-9]+)\n
