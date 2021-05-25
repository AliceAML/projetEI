from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import pickle
from scipy import sparse

# import example list
with open("examples_V2", "rb") as f:
    examples = pickle.load(f)

print(f"Loaded list of {len(examples[0])} examples")

y = examples[2]

with open("features_V2.npz", "rb") as f:
    X = sparse.load_npz(f)
print(f"Loaded features with shape {X.shape}")


# STANDARDISATION DONNEES
# TODO

# divide test train
print("TRAIN-TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

model = SVC(verbose=True, max_iter=1000)  # FIXME optimal class_weight ?
print("TRAINING")
model.fit(X_train, y_train)

print("EVALUATION")
predict_test = model.predict(X_test)
print("ACCURACY :", accuracy_score(y_test, predict_test))
print("RECALL :", recall_score(y_test, predict_test))
print(confusion_matrix(y_test, predict_test))


# TODO save model

# TODO implémenter grid_search


def false_negatives():
    pass


def false_positives():
    pass


def predict(sentence: str):
    # tokenize
    # make examples
    # vectorize
    # predict
    pass


def process(sentence: str):
    # predict
    # transform
    pass