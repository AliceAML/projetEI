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


with open("../corpus_no_ei_no_spacy.conll") as f, open(
    "../corpus_sans_ei_labelled.conll", "w"
) as g:
    content = f.read()
    sentences = content.split("\n\n")
    for sent in sentences:
        lines = sent.splitlines()
        tokens = {}
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
                # print(is_ei, is_ei == "True")
                tokens[id] = {"og": og, "is_ei": (is_ei == "True"), "no_ei": no_ei}
        doc = nlp(text)
        conll = doc._.conll_pd

        eis = []
        for i in range(len(conll)):
            if tokens[str(i)]["is_ei"]:
                eis.append(f"ei={tokens[str(i)]['og']}")
            else:
                eis.append("_")

        conll["misc"] = eis
        conll["id"] = conll.index

        ## printable version (for logs)
        # print(conll.to_string(index=None), end="\n\n")

        # version to add to the corpus (one tab between each value)
        conll_str = conll.to_csv(sep="\t", index=None)
        g.write(conll_str)
        g.write("\n\n")
