"""Main programme for "écriture inclusive" project.
Training, evaluation, prediction.
Alice HAMMEL & Marjolaine RAY"""

from email.mime import base
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import pickle
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import argparse
import conllu
import pandas as pd
import numpy as np

from tokenization import word_tokenize
from extract_features_spacy import nlp
from feat_vectorisation import (
    dummy_fun,
    make_examples_parsed_conll_sent,
    make_matrix,
    xpos_vectorizer,
    form_vectorizer,
    get_features,
)


WINDOW_SIZE = 2

# print("STANDARDIZE FEATURES")
# scaler = StandardScaler(with_mean=False)
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


def trainSVM(gridsearch=False):
    """Train SVM model
    Returns trained model."""  # TODO utiliser halving + augmenter max_iter
    if gridsearch:  # TODO add scaler to pipeline
        svc = SVC(verbose=True, max_iter=10000)
        parameters = {
            "C": [0.1, 1, 10, 100],
            "gamma": [1, 0.1, 0.01, 0.001],
            "kernel": ["rbf", "poly", "sigmoid"],
        }
        gs = GridSearchCV(estimator=svc, param_grid=parameters, scoring="f1", verbose=3)
        gs.fit(X_TRAIN, Y_TRAIN)
        print(f"Best parameters : {gs.best_params_}")
        print(f"Best f1 score : {gs.best_score_}")
        df = pd.Dataframe(gs.cv_results_)
        print(df)
        df.to_csv("gridsearch_results_SVM.csv", sep="\t")
        model = gs.best_estimator_
    else:
        model = SVC(verbose=True, max_iter=1000)

        print("TRAINING SVM MODEL")
        model.fit(X_TRAIN, Y_TRAIN)

    return model


def trainRandomForest(gridsearch=False):
    if gridsearch:
        random_forest = RandomForestClassifier(class_weight="balanced", verbose=1)
        parameters = {
            # "n_estimators": (50, 75, 100, 150),
            # "max_depth": (10, 15, 20),
            "n_estimators": (150, 175, 200),
            "max_depth": (20, 25, 30)
            # "min_samples_split": (1, 2, 4),
            # "max_features": (None, "auto", "sqrt", "log2"),
            # "class_weight": ("balanced", "balanced_subsample", None),
        }
        gs = GridSearchCV(
            estimator=random_forest, param_grid=parameters, scoring="f1", verbose=3
        )
        gs.fit(X_TRAIN, Y_TRAIN)
        print(f"Best parameters : {gs.best_params_}")
        print(f"Best f1 score : {gs.best_score_}")
        df = pd.Dataframe(gs.cv_results_)
        print(df)
        df.to_csv("gridsearch_results_SVM.csv", sep="\t")
        model = gs.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=75, max_depth=10, class_weight="balanced"
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
    print("F1 SCORE : ", f1_score(y, y_pred))
    print(confusion_matrix(y, y_pred))

    print("--BASELINE")
    if x is X_TEST:
        bl = baseline(EXAMPLES_TEST)
    else:
        bl = baseline(EXAMPLES_TRAIN)
    print("ACCURACY :", accuracy_score(y, bl))
    print("RECALL : ", recall_score(y, bl))
    print("PRECISION : ", precision_score(y, bl))
    print("F1 SCORE : ", f1_score(y, bl))
    print(confusion_matrix(y, bl))


def most_important_RF(model):
    """Prints the 20 most important features for this RandomForest Classifier"""
    assert isinstance(
        model, RandomForestClassifier
    ), "This model is not a RandomForestClassifier"
    order = np.argsort(model.feature_importances_)
    top = order[-50:][::-1]  # slice and reverse
    for i, feat in enumerate(top):
        if model.feature_importances_[feat] > 0:
            print(
                f"{i:03} - feat n°{feat:06} {get_features(feat):<25} importance: {model.feature_importances_[feat]}"
            )


def baseline(examples):
    """Predicts 1 if token is a NOUN (only on X)"""
    baseline_predict = []
    for ex in examples[0]:
        if (
            ex[0]["upos"] in ["NOUN", "PRON", "ADJ"]
            or "Tense=Past|VerbForm=Part" in ex[0]["xpos"]
        ):
            baseline_predict.append(1)
        else:
            baseline_predict.append(0)

    return baseline_predict


def false_negatives(model, x, y, examples):
    """Print ALL false negatives in X (TRAIN + TEST !)"""
    y_pred_X = model.predict(x)
    for i, gold in enumerate(y):
        if gold == 1 and y_pred_X[i] == 0:
            pprint_example(examples, i)
            print_example(examples, i)
            print()


def false_positives(model, x, y, examples):
    """Print ALL false positives in X (TRAIN + TEST !)"""
    y_pred_X = model.predict(x)
    for i, gold in enumerate(y):
        if gold == 0 and y_pred_X[i] == 1:
            pprint_example(examples, i)
            print_example(examples, i)
            print()


def pprint_example(examples, i: int):
    """Print token + context as a string"""
    context_forms = [x["form"] for x in examples[1][i]]
    begin = context_forms[0:WINDOW_SIZE]  # récup forms
    token = examples[0][i][0]["form"]
    end = context_forms[WINDOW_SIZE:]  # récup forms
    output = " ".join(begin + ["_"] + [token] + ["_"] + end)
    print(output)


def print_example(examples, i):
    # print("gold_label :", examples[3][i])  # label
    print("token :", examples[0][i])  # token
    print("context :", examples[1][i])  # context
    print("sentence :", examples[2][i])


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
    # TODO transformation phrase
    pass


###############################################################################""
parser = argparse.ArgumentParser(
    description="Automatically transform sentences into inclusive writing (French)."
)
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
    help='display evaluation metrics for current model. Choose "all" for baseline.',
    choices=["test", "train"],
)
parser.add_argument(
    "--train", choices=["RandomForest", "SVM"], help="Choose a model to train."
)
parser.add_argument(
    "--gridsearch",
    type=bool,
    help="search best parameters during training with a grid search",
    default=False,
)
parser.add_argument(
    "--save", help="Save the model with a chosen version number.", type=int
)


args = parser.parse_args()
###############################################################


with open("examples_V5", "rb") as f:  # EXAMPLE VERSION
    EXAMPLES = pickle.load(f)

print(f"Loaded list of {len(EXAMPLES[0])} examples")

Y = EXAMPLES[3]

with open("features_V5.npz", "rb") as f:  # CORRESPONDING FEATURES
    X = sparse.load_npz(f)
print(f"Loaded features with shape {X.shape}")

# divide test train
print("TRAIN-TEST SPLIT (80\%-20\%)")
EXAMPLES_TRAIN = [None] * 3
EXAMPLES_TEST = [None] * 3

(
    X_TRAIN,
    X_TEST,
    Y_TRAIN,
    Y_TEST,
    EXAMPLES_TRAIN[0],  # tokens
    EXAMPLES_TEST[0],
    EXAMPLES_TRAIN[1],  # contexts
    EXAMPLES_TEST[1],
    EXAMPLES_TRAIN[2],  # sentences
    EXAMPLES_TEST[2],
) = train_test_split(
    X,
    Y,
    EXAMPLES[0],  # tokens
    EXAMPLES[1],  # contexts
    EXAMPLES[2],  # sentences
    test_size=0.2,
    random_state=8,
)


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
    print("Evaluation on TEST")
    eval(model, X_TEST, Y_TEST)
elif args.eval == "train":
    print("Evaluation on TRAIN")
    eval(model, X_TRAIN, Y_TRAIN)

if args.convert:
    prediction = predict_sentence(args.convert)
    print(args.convert)
    print(*list(prediction))

if args.train:
    if args.train == "RandomForest":
        model = trainRandomForest(gridsearch=args.gridsearch)
    if args.train == "SVM":
        model = trainSVM(gridsearch=args.gridsearch)
    print("EVALUATION ON TEST")
    eval(model, X_TEST, Y_TEST)


if args.save:
    save_model(model, args.save)