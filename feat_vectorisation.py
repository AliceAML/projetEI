#%%
from conllu import parse_incr, parse
from sklearn.feature_extraction.text import CountVectorizer

WINDOW_SIZE = 2

# http://www.davidsbatista.net//blog/2018/02/28/TfidfVectorizer/
# dummy function to use pretokenized data
def dummy_fun(doc):
    return doc


form_vectorizer = CountVectorizer(
    analyzer="word", tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None
)

xpos_vectorizer = CountVectorizer(
    analyzer="word", tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None
)

#%% CREATION OF FORM VECTORIZER AND XPOS VECTORIZER
with open("../3.1.corpus_spacied.conll", "r") as data_file:

    def form_generator():
        for sent in parse_incr(data_file):
            yield [tok["form"] for tok in sent]

    form_vectorizer.fit(form_generator())
    print(form_vectorizer.vocabulary_)

# re-opening the file to rset the parse_incr generator
with open("3.1.corpus_spacied.conll", "r") as data_file:

    def xpos_generator():
        for sent in parse_incr(data_file):
            for tok in sent:
                yield (tok["xpos"].split("|"))

    xpos_vectorizer.fit(xpos_generator())
    print(xpos_vectorizer.vocabulary_)


#%% GENERATING THE EXAMPLES
test = """# sent_id = 344
# text_no_ei =  J ’ étais très coupée de mes amis .
0	J	j	NOUN	NOUN	_	2	nummod	_	_
1	’	’	PUNCT	PUNCT	_	3	nsubj	_	_
2	étais	étai	ADJ	ADJ__Gender=Masc	_	0	ROOT	_	_
3	très	très	ADV	ADV	_	5	advmod	_	_
4	coupée	couper	VERB	VERB__Gender=Fem|Number=Sing|Tense=Past|VerbForm=Part	_	3	obj	_	_
5	de	de	ADP	ADP	_	8	case	_	_
6	mes	mon	DET	DET__Number=Plur|Poss=Yes	_	8	det	_	_
7	amis	ami	NOUN	NOUN__Gender=Masc|Number=Plur	_	5	nmod	_	ei=ami-e-s
8	.	.	PUNCT	PUNCT	_	3	punct	_	_
"""
test_data = parse(test)

# modifier pour que les fonctions prennent plusieurs "phrases" en même temps ?

"""Takes in a all the examples 
and returns their xpos vectors"""


def xpos_vectorize(tokenlists):
    # build a list of xpos
    xpos = []
    for tokenlist in tokenlists:
        xpos_list = []
        for tok in tokenlist:
            for feat in tok["xpos"].split("|"):
                xpos_list.append(feat)
        xpos.append(xpos_list)
    return xpos_vectorizer.transform(xpos)


"""Takes in a all the examples 
and returns their form vectors"""


def form_vectorize(tokenlists):
    form_lists = []
    for tokenlist in tokenlists:
        form_lists.append([tok["form"] for tok in tokenlist])
    return form_vectorizer.transform(form_lists)


# forme vecteur
# WORD form _ WORD xpos _ CONTEXT forms _ CONTEXT xpos _ LABEL


"""Takes in a all the examples for 1 sentence 
and returns their complete vectors with form, pos and label"""
# TODO : comment trouver le mot example pour chaque contexte ?
# fournir une liste du type [(w,cont,label), (w,cont,label)]...
# renvoyer [vecteur1, vecteur2]


#%% SEPARATION TRAIN/DEV/TEST TODO