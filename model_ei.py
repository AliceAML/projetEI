"""Train models, evaluate, predict."""

from ast import arg
from sklearn.model_selection import train_test_split
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

WINDOW_SIZE = 2

# import example list
with open("examples_V2", "rb") as f:
    EXAMPLES = pickle.load(f)

print(f"Loaded list of {len(EXAMPLES[0])} examples")

Y = EXAMPLES[2]

with open("features_V2.npz", "rb") as f:
    X = sparse.load_npz(f)
print(f"Loaded features with shape {X.shape}")


# divide test train
print("TRAIN-TEST SPLIT (80\%-20\%)")
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2, random_state=8)

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
        n_estimators=100, max_depth=15, verbose=3, class_weight="balanced"
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
    print(f"PICKLING TO {filename}")
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
    print("gold_label :", EXAMPLES[2][i])  # label
    print("token :", EXAMPLES[0][i])  # token
    print("context :", EXAMPLES[1][i])  # context


def predict_sentence(sentence: str):
    # tokenize
    # make examples
    # vectorize
    # predict
    pass


def process_sentence(sentence: str):
    # predict
    # transform
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
    "--eval", type=str, help="evaluate the model", choices=["test", "train"]
)
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
    model = load_model("RandomForestClassifier_V1")


if args.eval == "test":
    eval(model, X_TEST, Y_TEST)
elif args.eval == "train":
    eval(model, X_TEST, Y_TEST)
