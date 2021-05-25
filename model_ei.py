from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import pickle
from scipy import sparse
from sklearn.preprocessing import StandardScaler

# import example list
with open("examples_V2", "rb") as f:
    examples = pickle.load(f)

print(f"Loaded list of {len(examples[0])} examples")

y = examples[2]

with open("features_V2.npz", "rb") as f:
    X = sparse.load_npz(f)
print(f"Loaded features with shape {X.shape}")


# divide test train
print("TRAIN-TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# print("STANDARDIZE FEATURES")
# scaler = StandardScaler(with_mean=False)
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

model = SVC(verbose=True, max_iter=1000)  # FIXME optimal class_weight ?


print("TRAINING")
model.fit(X_train, y_train)

print("----RESULTS OF TRAINING")
y_pred_train = model.predict(X_train)
print("ACCURACY :", accuracy_score(y_train, y_pred_train))
print("RECALL : ", recall_score(y_train, y_pred_train))
print("PRECISION : ", precision_score(y_train, y_pred_train))
print(confusion_matrix(y_train, y_pred_train, labels=[0, 1]))

print("-----EVALUATION ON TEST DATASET")
y_pred = model.predict(X_test)
print("ACCURACY :", accuracy_score(y_test, y_pred))
print("RECALL : ", recall_score(y_test, y_pred))
print("PRECISION : ", precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("---- EVALUATION ON ALL EXAMPLES X")
print("for analysis...")
y_pred_X = model.predict(X)

with open("model_V1", "wb") as f:
    pickle.dump(model, f)


# TODO save model

# TODO implémenter grid_search

# A UTILISER UNIQUEMENT avec X
def false_negatives():
    for i, gold in enumerate(y):
        if gold == 1 and y_pred_X[i] == 0:
            print(examples[2][i])
            print(examples[0][i])
            print(examples[1][i])
            print()


def false_positives():
    for i, gold in enumerate(y):
        if gold == 0 and y_pred_X[i] == 1:
            print(examples[2][i])
            print(examples[0][i])
            print(examples[1][i])
            print()


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