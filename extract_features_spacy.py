"""Extracts the features from non_EI sentences,
and outputs them to a conll-U file,
with the ei labels in the misc column.

Returns:
    conll file: conll-U file,
with the ei labels in the misc column.
"""

import spacy
from spacy_conll import ConllFormatter
from spacy.tokens import Doc
import pandas as pd

pd.options.display.max_columns = 10

nlp = spacy.load("fr_core_news_sm")


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

# conversion map pour garder colonne MISC vide pour les EI
conllformater = ConllFormatter(nlp, conversion_maps={"misc": {"SpaceAfter=No": "_"}})
nlp.add_pipe(conllformater)


if __name__ == "__main__":
    with open("2.corpus_no_ei.conll") as f, open("3.1.corpus_spacied.conll", "w") as g:
        content = f.read()
        sentences = content.split("\n\n")
        for sent in sentences[:-1]:
            lines = sent.splitlines()
            eis = []
            for line in lines:
                if line.startswith("# doc path"):
                    g.write(line + "\n")
                    print(line)
                elif line.startswith("# sent_id"):
                    g.write(line + "\n")
                elif line.startswith("# text_no_ei"):
                    g.write(line + "\n")
                    _, text = line.split(" =  ")
                elif line.startswith(tuple("0123456789")):
                    id, og, is_ei, no_ei = line.split("\t")
                    if is_ei == "True":
                        eis.append(f"ei={og}")
                    else:
                        eis.append("_")
            doc = nlp(text)
            conll = doc._.conll_pd

            conll["misc"] = eis
            conll["id"] = conll.index

            ## printable version (for logs)
            # print(conll.to_string(index=None), end="\n\n")

            # version to add to the corpus (one tab between each value)
            conll_str = conll.to_csv(sep="\t", index=None, header=False)
            g.write(conll_str)
            g.write("\n\n")
