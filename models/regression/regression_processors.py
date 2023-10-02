from sklearn.model_selection import train_test_split
from itertools import product
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression


class RegressionModels:
    def __init__(self, model_class_names, configs) -> None:
        self.model_class_names = model_class_names
        self.configs = configs

        self._execute()

    def _execute(self):
        # Import necessary libraries

        model_data_df = pd.read_csv(self.configs["model_data"])
        X = model_data_df[self.configs["independent_variables"]].values
        y = model_data_df[self.configs["dependent_variable"]].values

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create a linear regression model
        model = LinearRegression()

        # Train the model on the training set
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        test_score = model.score(X_test, y_test)
        print("Test R^2 Score:", test_score)

        # Perform cross-validation on the training set
        cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        # Print the cross-validation scores on the training set
        print("Cross-Validation Scores on Training Set:", cross_val_scores)
        print("Mean R^2 Score on Training Set:", np.mean(cross_val_scores))

        print("Inside Regression class")
