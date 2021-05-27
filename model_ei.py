"""Main programme for "écriture inclusive" project.
Training, evaluation, prediction.
Alice HAMMEL & Marjolaine RAY"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import pickle
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import argparse
import conllu

from tokenization import word_tokenize
from extract_features_spacy import nlp
from feat_vectorisation import dummy_fun, make_examples_parsed_conll_sent, make_matrix


WINDOW_SIZE = 2

# print("STANDARDIZE FEATURES")
# scaler = StandardScaler(with_mean=False)
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


def trainSVM():
    """Train SVM model
    Returns trained model."""
    model = SVC(verbose=True, max_iter=1000)

    print("TRAINING SVM MODEL")
    model.fit(X_TRAIN, Y_TRAIN)

    return model


def trainRandomForest():
    model = RandomForestClassifier(
        n_estimators=75, max_depth=10, verbose=3, class_weight="balanced"
    )

    model.fit(X_TRAIN, Y_TRAIN)

    return model


def save_model(model, version_num):
    """Save model to a file (pickle)
    filename format : {type_model}_V{version_num}

    Args:
        model (sklearn model): trained model
        version_num ([type]): version number to differentiate from previous stored models.
    """
    type_model = str(type(model)).split(".")[-1].strip("'>")
    filename = f"{type_model}_V{version_num}"
    print(f"PICKLING MODEL TO {filename}")
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(file: str):
    with open(file, "rb") as f:
        return pickle.load(f)


def eval(model, x, y):
    """
    Use model to classify examples from x and compare to Y.
    Prints confusion matrix, accuracy, recall and precision.
    """
    print(model)
    y_pred = model.predict(x)
    print("ACCURACY :", accuracy_score(y, y_pred))
    print("RECALL : ", recall_score(y, y_pred))
    print("PRECISION : ", precision_score(y, y_pred))
    print(confusion_matrix(y, y_pred))


# TODO implémenter grid_search


def false_negatives(model):
    """Print ALL false negatives in X (TRAIN + TEST !)"""
    y_pred_X = model.predict(X)
    for i, gold in enumerate(Y):
        if gold == 1 and y_pred_X[i] == 0:
            pprint_example(i)
            print_example(i)
            print()


def false_positives(model):
    """Print ALL false positives in X (TRAIN + TEST !)"""
    y_pred_X = model.predict(X)
    for i, gold in enumerate(Y):
        if gold == 0 and y_pred_X[i] == 1:
            pprint_example(i)
            print_example(i)
            print()


def pprint_example(i: int):
    """Print token + context as a string"""
    context_forms = [x["form"] for x in EXAMPLES[1][i]]
    begin = context_forms[0:WINDOW_SIZE]  # récup forms
    token = EXAMPLES[0][i][0]["form"]
    end = context_forms[WINDOW_SIZE:]  # récup forms
    output = " ".join(begin + ["_"] + [token] + ["_"] + end)
    print(output)


def print_example(i):
    print("gold_label :", EXAMPLES[3][i])  # label
    print("sentence :", EXAMPLES[2][i])
    print("token :", EXAMPLES[0][i])  # token
    print("context :", EXAMPLES[1][i])  # context


def predict_sentence(sentence: str):
    # tokenize
    tokens = word_tokenize(sentence)
    # get spacy tags + connlize
    doc = nlp(" ".join(tokens))
    conll = doc._.conll_pd.to_csv(sep="\t", index=None, header=False)
    # make examples
    dico_conll = conllu.parse(conll)
    examples = make_examples_parsed_conll_sent(dico_conll[0])
    # vectorize
    feats = make_matrix(examples)
    # # predict
    y_pred = model.predict(feats)
    return zip(tokens, y_pred)


def process_sentence(sentence: str):
    # predict
    pred = predict_sentence(sentence)
    # transform
    # TODO
    pass


parser = argparse.ArgumentParser()
parser.add_argument(
    "--convert", type=str, help="convert the given string into écriture inclusive."
)
parser.add_argument(
    "--load",
    type=str,
    help="load model from a file. By default the model used is the one with the best results.",
)
parser.add_argument(
    "--eval",
    type=str,
    help="display evaluation metrics for current model",
    choices=["test", "train"],
)
parser.add_argument(
    "--train", choices=["RandomForest", "SVM"], help="Choose a model to train."
)
parser.add_argument(
    "--save", help="Save the model with a chosen version number.", type=int
)
# parser.add_argument("--verbose", type=bool)
# parser.add_argument(
#     "--model", type=str, choices=["SVM", "RandomForest"], help="choose model"
# )

args = parser.parse_args()

if args.load:
    print(f"Loading {args.load} ....")
    try:
        model = load_model(args.load)
    except Exception as e:
        print(e)
    print("Model loaded")
else:
    model = load_model("RandomForestClassifier_V2")  # DEFAULT MODEL

if args.eval == "test":
    eval(model, X_TEST, Y_TEST)
elif args.eval == "train":
    eval(model, X_TRAIN, Y_TRAIN)

if args.convert:
    prediction = predict_sentence(args.convert)
    print(args.convert)
    print(*list(prediction))

with open("examples_V5", "rb") as f:  # EXAMPLE VERSION
    EXAMPLES = pickle.load(f)

print(f"Loaded list of {len(EXAMPLES[0])} examples")

Y = EXAMPLES[3]

with open("features_V5.npz", "rb") as f:  # CORRESPONDING FEATURES
    X = sparse.load_npz(f)
print(f"Loaded features with shape {X.shape}")

# divide test train
print("TRAIN-TEST SPLIT (80\%-20\%)")
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2, random_state=8)
if args.train == "RandomForest":
    model = trainRandomForest()
if args.train == "SVM":
    model = trainSVM()

if args.train:
    # import example list

    eval(model, X_TEST, Y_TEST)

if args.save:
    save_model(model, args.save)