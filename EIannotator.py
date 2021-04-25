"""
Annotates texts to indicate which words are EI forms.
"""

import sys
import os

sys.path.append("../codeEI")

import regex as re
from findEI import isEI
from datetime import date


# print("text 100", text[0:100])
# à propos de la regex : \b pour identifier les fins de mots dans une string (utilisable sur une pharse, il faut qu'il y ai
# un séparateur. si on passe sur des items : pas de sep)
# print("sentences 10", *sentences[:10], sep="\n\n")


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as data:
        text = data.read()
        clean = text.replace("\t", " ").replace("\n", " ")
        # output = clean.lower()
        output = clean

    return output


def word_tokenize(sentence: str) -> list:
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


def sent_tokenize(text: str) -> list:
    return re.split(r"(?<=\.|\?|\!)\s", text)


def labelling(sentences: list) -> list:
    """
    Label words in a list of sentences (EI or not EI)
    Returns a list of annotated sentences.
    """
    out = []
    for one_sentence in sentences:
        tempwords = word_tokenize(one_sentence)
        tempgold = []
        for index, word in enumerate(tempwords):
            if isEI(word):
                tempgold.append(index)
        tempdict = dict()
        tempdict["words"] = tempwords
        tempdict["gold"] = tempgold
        out.append(tempdict)
    return out


def CoNLL_label(filepath):
    """Finds EI forms and annotate file in CoNLL format"""
    # https://stanfordnlp.github.io/stanza/data_conversion.html#python-object-to-conll
    out = f"# doc path = {filepath}\n"
    text = load_file(filepath)
    sentences = sent_tokenize(text)
    for i, sent in enumerate(sentences):
        out += f"# sent_id = {i}\n"
        out += f"# text = {sent}\n"
        words = word_tokenize(sent)
        for j, word in enumerate(words):
            out += f"{j}\t{word}\t{isEI(word)}\n"
        out += "\n"
    return out


# def label_file(filepath):
#     name = filepath
#     text = load_file(name)
#     sentences = re.split(r"(?<=\.|\?|\!)\s", text)

#     labels = labelling(sentences)


# output = list(map(lambda x: x.replace('\t','').replace('\n',''),text))

# 1  nettoyer le texte (tabulation? passages de lignes?)
#       phrases coupees par des sauts de lignes = ok
#       phrases de titres : elles n'ont pas de points, ça les fusionne
# 2  mettre en forme de corpus (par textes entiers?)
# 3  estimer les meilleures regex

# \n([a-zA-Z0-9]+|[A-Z]\+[a-zA-Z0-9]+)\n

# if script launched as main
if __name__ == "__main__":
    if sys.argv[1] == "all":
        path = "corpus/infokiosque"
        today = date.today().strftime("%y%m%d")
        with open(f"infokiosque_labelled.conll", "w") as f:
            for file in os.listdir(path):
                print(file)
                f.write(CoNLL_label(path + "/" + file))

    else:
        name = sys.argv[1]

        text = load_file(name)
        sentences = re.split(r"(?<=\.|\?|\!)\s", text)

        labels = labelling(sentences)

        if len(sys.argv) > 2 and sys.argv[2] == "-v":
            print(labels)

        for i in labels:
            if i["gold"]:
                for j in i["gold"]:
                    print(i["words"][j])
