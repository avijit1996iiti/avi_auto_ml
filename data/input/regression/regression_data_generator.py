import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target
X = pd.DataFrame(X)
y = pd.DataFrame(y, columns=["target"])
data_df = pd.concat([X, y], axis=1)
data_df.to_csv("data/input/regression/regression_dummy_data.csv", index=None)
