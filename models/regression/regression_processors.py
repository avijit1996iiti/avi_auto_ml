from sklearn.model_selection import train_test_split
from itertools import product
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score


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
            X,
            y,
            test_size=self.configs["train_test_split"]["test_size"],
            random_state=42,
        )
        for model_name in self.configs["models_to_run"].keys():
            # Step 5: Define Hyperparameter Grid
            param_grid = self.configs["models_to_run"][model_name]

            # Step 6: Generate all possible combinations of hyperparameters
            param_combinations = list(product(*param_grid.values()))

            # perform grid search with logging every information in mlflow
            for params_dict in pd.DataFrame(
                param_combinations, columns=param_grid.keys()
            ).to_dict("r"):
                with mlflow.start_run():
                    # Create and train the model with the current hyperparameters
                    model = self.model_class_names[self.configs["model_type"]][
                        model_name
                    ](
                        **params_dict
                    )  # You can use any scikit-learn model
                    # todo : add cross validation
                    model.fit(X_train, y_train)
                    # Log the hyperparameters
                    mlflow.log_params(params_dict)
                    mlflow.log_param("model_name", model_name)
                    # Log evaluation metrics (e.g., accuracy, F1-score)
                    # Evaluate the model on the test set
                    test_score = model.score(X_test, y_test)
                    mlflow.log_metric("R_2", test_score)

                    # Log the trained model as an artifact
                    mlflow.sklearn.log_model(model, "model")
                    # todo : add model signature
                    # todo : add all test metrics
                    # todo : add feature of regression
                    """
                    # Perform cross-validation on the training set
                    cross_val_scores = cross_val_score(
                        model, X_train, y_train, cv=5, scoring="r2"
                    )

                    # Print the cross-validation scores on the training set
                    print("Cross-Validation Scores on Training Set:", cross_val_scores)
                    print("Mean R^2 Score on Training Set:", np.mean(cross_val_scores))

                    print("Inside Regression class")
                    """
