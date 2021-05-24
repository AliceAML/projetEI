#%%
from bleach import VERSION
from conllu import parse_incr, parse
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from scipy import sparse
import sys

FILE = "3.1.corpus_spacied.conll"
WINDOW_SIZE = 2
f1 = lambda x: {"form": f"d{x}", "xpos": None}
f2 = lambda x: {"form": f"f{x}", "xpos": None}
FAKE_WORDS = [f(i) for i in range(WINDOW_SIZE) for f in (f1, f2)]

VERSION_NB = 1


# http://www.davidsbatista.net//blog/2018/02/28/TfidfVectorizer/
# dummy function to use pretokenized data
def dummy_fun(doc):
    return doc


#%% CREATION OF FORM VECTORIZER AND XPOS VECTORIZER


def form_generator():

    # iniitalise with fake words for beginning and end (depending on WINDOW_SIZE)
    yield from FAKE_WORDS
    with open(FILE, "r") as data_file:
        for sent in parse_incr(data_file):
            yield [tok["form"] for tok in sent]


def xpos_generator():
    with open(FILE, "r") as data_file:
        for sent in parse_incr(data_file):
            for tok in sent:
                yield (tok["xpos"].split("|"))


#%% GENERATING THE EXAMPLES


"""Takes in a list of words 
and returns their xpos vectors"""


def xpos_vectorize(tokenlists):
    # build a list of xpos
    xpos = []
    for tokenlist in tokenlists:
        xpos_list = []
        for tok in tokenlist:
            if tok["xpos"]:
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
and returns the full matrix with form, pos and label"""
# input : [[w1, w2, w3,...], [cont1, cont2, ...], [label1, label2,...]]
# renvoyer [vecteur1, vecteur2] (matrice de vecteurs)
def make_matrix(examples, labels=True):
    if labels:
        words, contexts, gold_labels = examples
        gold = sparse.csr_matrix(gold_labels)
    else:
        words, contexts = examples

    word_form = form_vectorize(words)
    word_xpos = xpos_vectorize(words)
    context_forms = form_vectorize(contexts)
    context_xpos = xpos_vectorize(contexts)

    if labels:
        return sparse.hstack(
            [word_form, word_xpos, context_forms, context_xpos, gold.transpose()]
        )
    else:
        return sparse.hstack([word_form, word_xpos, context_forms, context_xpos])


"""Make a list of examples from a CONLL file
[[w1, w2, w3,...], [cont1, cont2, ...], [label1, label2,...]]"""


def make_examples(file, labels=True):
    with open(file) as data_file:
        words = []
        contexts = []
        gold_labels = []
        for sent in parse_incr(data_file):
            words += [
                [tok] for tok in sent
            ]  # adding all the tokens to the example list
            # adding fake words to the sentence
            with_fakes = FAKE_WORDS[0::2] + sent + FAKE_WORDS[1::2]

            # embedded list comprehension that generates a list of context tokens for each word of the sentence
            contexts += [
                [
                    with_fakes[j]
                    for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1)
                    if j != i  # without the word itself
                ]
                for i in range(WINDOW_SIZE, len(with_fakes) - WINDOW_SIZE)
            ]

            def is_ei(misc):
                if misc:
                    return 1
                else:
                    return 0

            gold_labels += [is_ei(x["misc"]) for x in sent]
    return [words, contexts, gold_labels]


#%% MAIN
form_vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None,
)

xpos_vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None,
)

try:
    print("UNPICKLING VECTORIZERS")
    with open("vectorizers_V1", "rb") as f:
        form_vectorizer = pickle.load(f)
        xpos_vectorizer = pickle.load(f)
except Exception as e:
    print(e)

    print("FIT FORMS VECTORIZER")
    form_vectorizer.fit(form_generator())
    print(form_vectorizer.vocabulary_)

    print("FIT XPOS VECTORIZER")
    xpos_vectorizer.fit(xpos_generator())
    print(xpos_vectorizer.vocabulary_)

try:
    print("UNPICKLING EXAMPLES")
    with open("examples_V1", "rb") as f:
        examples = pickle.load(f)
except Exception as e:
    print(e)
    print("GENERATING EXAMPLES")
    examples = make_examples(FILE, labels=True)

file_vectorizers = f"vectorizers_V{VERSION_NB}"
with open(file_vectorizers, "wb") as f:
    print(f"PICKLING FORMS VECTORIZER to {file_vectorizers}")
    pickle.dump(form_vectorizer, f)

    print(f"PICKLING XPOS VECTORIZER to {file_vectorizers}")
    pickle.dump(xpos_vectorizer, f)

file_examples = f"examples_V{VERSION_NB}"
print(f"PICKLING EXAMPLES to {file_examples}")
with open(file_examples, "wb") as f:
    pickle.dump(examples, f)


print("CONVERTING EXAMPLES TO A MATRIX")
feat_matrix = make_matrix(examples, labels=True)

file_matrix = f"features_V{VERSION_NB}"
print(f"SAVING MATRIX to {file_matrix}")
sparse.save_npz(file_matrix, feat_matrix, compressed=True)


#%% SEPARATION TRAIN/DEV/TEST TODO --> à faire dans MODELE SVM ?
