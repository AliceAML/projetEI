#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import sparse
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#%% load vectorized examples
with open("../features_V5.npz", "rb") as f:  # CORRESPONDING FEATURES
    X = sparse.load_npz(f)
print(f"Loaded features with shape {X.shape}")

#%% load text examples + labels
with open("../examples_V5", "rb") as f:  # EXAMPLE VERSION
    EXAMPLES = pickle.load(f)
print("loaded examples")
Y = EXAMPLES[3]

#%% reduce to 2 dimensions with TruncatedSVD
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("svd", TruncatedSVD(n_components=2)),
    ]
)
X_svd = pipeline.fit_transform(X)
svd_df = pd.DataFrame(columns=["C1", "C2"], data=X_svd)
svd_df["label"] = Y


#%% visualize svd_df
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("C1", fontsize=15)
ax.set_ylabel("C2", fontsize=15)
ax.set_title("2 component SVD", fontsize=20)
targets = [0, 1]
colors = ["r", "b"]
labels = {0: "Non-EI", 1: "EI"}
for target, color in zip(targets, colors):
    target_indices = svd_df["label"] == target
    svd_df.loc[target_indices, "C1"]
    svd_df.loc[target_indices, "C2"]
    ax.scatter(
        svd_df.loc[target_indices, "C1"],
        svd_df.loc[target_indices, "C2"],
        c=color,
        s=50,
        label=labels[target],
    )
ax.legend()
ax.grid()
plt.show()
