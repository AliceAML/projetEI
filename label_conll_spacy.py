import spacy
from spacy_conll import ConllFormatter
import pandas as pd

pd.options.display.max_columns = 10

nlp = spacy.load("fr_core_news_sm")

# conversion map pour garder colonne MISC vide pour les EI
conllformater = ConllFormatter(nlp, conversion_maps={"misc": {"SpaceAfter=No": "_"}})
nlp.add_pipe(conllformater)

test_corpus = """# sent_id = 5
# text = C’est d’ailleurs appâté·e·s par ces clichés que nous sommes venu·e·s sur la ZAD y voir de plus près.
0	C	False	C
1	’	False	’
2	est	False	est
3	d	False	d
4	’	False	’
5	ailleurs	False	ailleurs
6	appâté·e·s	True	appâtés
7	par	False	par
8	ces	False	ces
9	clichés	False	clichés
10	que	False	que
11	nous	False	nous
12	sommes	False	sommes
13	venu·e·s	True	venus
14	sur	False	sur
15	la	False	la
16	ZAD	False	ZAD
17	y	False	y
18	voir	False	voir
19	de	False	de
20	plus	False	plus
21	près	False	près
22	.	False	.
# text_ei =  C ’ est d ’ ailleurs appâtés par ces clichés que nous sommes venus sur la ZAD y voir de plus près .
"""

# with open("corpus_ei_final.conll") as f, open("corpus_sans_ei_labelled.conll") as g:
# content = f.read()
# sentences = content.split("\n\n")
# for sent in sentences:
sent = test_corpus
lines = sent.splitlines()
tokens = {}
for line in lines:
    if line.startswith(("# doc path", "# sent_id")):
        # g.write(line)
        print(line)
    elif line.startswith("# text_ei"):
        # g.write(line)
        print(line)
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
print(conll.to_string(index=None))

# version to add to the corpus (one tab between each value)
conll_str = conll.to_csv(sep="\t")
