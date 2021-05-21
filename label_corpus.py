import spacy
from spacy_conll import ConllFormatter


nlp = spacy.load("fr_core_news_sm")
# conversion map pour garder colonne MISC vide pour les EI
conllformater = ConllFormatter(nlp, conversion_maps={"misc": {"SpaceAfter=No": "_"}})
nlp.add_pipe(conllformater)
with open("../corpus/corpus_ei/infokiosque_42") as f:
    doc = nlp(f.read().replace("\n", " "))

# conll = doc._.conll_pd
# print(conll)

for sent in doc.sents:
    print(sent)
    print(sent._.conll_pd)