"""
Annotates texts to indicate which words are EI forms.
"""
#%%
import sys
import os

sys.path.append("../codeEI")

from datetime import date


from findEI import isEI
from tokenization import *

#%%
# print("text 100", text[0:100])
# à propos de la regex : \b pour identifier les fins de mots dans une string (utilisable sur une pharse, il faut qu'il y ai
# un séparateur. si on passe sur des items : pas de sep)
# print("sentences 10", *sentences[:10], sep="\n\n")


def load_file(filename) -> str:
    with open(filename, "r", encoding="utf-8") as data:
        text = data.read()
        clean = text.replace("\t", " ").replace("\n", " ")
        # output = clean.lower()
        output = clean

    return output


#  == si mot non nul


#%%
def labelling(file: str) -> list:
    """
    Label words in a list of sentences (EI or not EI)
    Returns a list of annotated sentences.
    """
    sentences = sent_tokenize(load_file(file))
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
            out += f"{j:10}\t{word}\t{isEI(word)}\n"
        out += "\n"
    return out


def get_ei_from_labels(labels):
    """Takes in labels outputted by labelling()
    and returns a list of the EI forms (for auditing).

    Args:
        labels (list): list of labels outputted by labelling()
    """
    ei = []
    for i in labels:
        if i["gold"]:
            for j in i["gold"]:
                ei.append(i["words"][j])

    return ei


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
    if sys.argv[1] == "all":  # python3 EIannotator.py all
        path = "../corpus/corpus_ei"
        today = date.today().strftime("%y%m%d")
        new_path = "../corpus/corpus_ei"

        with open(f"corpus_ei_labelled.conll", "w") as f:
            for i, file in enumerate(os.listdir(path)):
                print(f"{i} / {len(os.listdir(path))}  - {file} ")
                f.write(CoNLL_label(path + "/" + file))

        # nb_docs_ei = 0

        # for file in os.listdir(path):
        #     if file not in os.listdir(new_path):
        #         print(file)
        #         try:
        #             labels = labelling(path + "/" + file)
        #             eis = get_ei_from_labels(labels)
        #             if eis:
        #                 print("---", file)
        #                 os.system(f"cp {path}/{file} {new_path}/{file}")
        #                 nb_docs_ei += 1
        #                 for ei in eis:
        #                     print(ei)
        #         except Exception as e:
        #             print(e)

        # print(f"{nb_docs_ei} / {len(os.listdir(path))}")
        # 7/05 10h 892 / 1387 (infokiosque only)
        # sans les compounds 647 / 1387 !

    else:
        name = sys.argv[1]

        labels = labelling(name)

        if len(sys.argv) > 2 and sys.argv[2] == "-v":
            print(labels)

        for i in labels:
            if i["gold"]:
                for j in i["gold"]:
                    print(i["words"][j])
