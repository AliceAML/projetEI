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

# import example list
with open("examples_V2", "rb") as f:
    EXAMPLES = pickle.load(f)

print(f"Loaded list of {len(EXAMPLES[0])} examples")

Y = EXAMPLES[2]

with open("features_V2.npz", "rb") as f:
    X = sparse.load_npz(f)
print(f"Loaded features with shape {X.shape}")


# divide test train
print("TRAIN-TEST SPLIT")
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

    Args:
        model (sklearn model): trained model
        version_num ([type]): version number to differentiate from previous stored models.
    """
    type_model = str(type(model)).split(".")[4]
    filename = f"{type_model}_V{version_num}"
    print(f"PICKLING TO {filename}")
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def eval(model, x, y):
    """
    Use model to classify examples from x and compare to Y.
    Prints confusion matrix, accuracy, recall and precision.
    """
    y_pred = model.predict(x)
    print("ACCURACY :", accuracy_score(y, y_pred))
    print("RECALL : ", recall_score(y, y_pred))
    print("PRECISION : ", precision_score(y, y_pred))
    print(confusion_matrix(y, y_pred))


# TODO implémenter grid_search

# A UTILISER UNIQUEMENT avec X
def false_negatives(model):
    y_pred_X = model.predict(X)
    # FIXME pprint sentence
    for i, gold in enumerate(Y):
        if gold == 1 and y_pred_X[i] == 0:
            print(EXAMPLES[2][i])
            print(EXAMPLES[0][i])
            print(EXAMPLES[1][i])
            print()


def false_positives(model):
    y_pred_X = model.predict(X)
    for i, gold in enumerate(Y):
        if gold == 0 and y_pred_X[i] == 1:
            print(EXAMPLES[2][i])
            print(EXAMPLES[0][i])
            print(EXAMPLES[1][i])
            print()


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