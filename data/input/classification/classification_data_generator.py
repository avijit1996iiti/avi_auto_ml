import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42,
)

X = pd.DataFrame(X)
y = pd.DataFrame(y, columns=["target"])
data_df = pd.concat([X, y], axis=1)
# data_df.to_csv("classification_dummy_data.csv", index=None)
print(data_df[[0, 1]].head().values)
