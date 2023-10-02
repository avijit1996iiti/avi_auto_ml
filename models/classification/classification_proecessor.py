from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from itertools import product
import mlflow
import pandas as pd
from sklearn.datasets import make_classification


class ClassificationModels:
    def __init__(self, model_class_names, configs) -> None:
        self.model_class_names = model_class_names
        self.configs = configs

        self._execute()

    def _execute(self):
        # read data from source
        model_data_df = pd.read_csv(self.configs["model_data"])
        X = model_data_df[self.configs["independent_variables"]].values
        y = model_data_df[self.configs["dependent_variable"]].values

        # Split the Data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.configs["train_test_split"]["test_size"],
            random_state=42,
        )

        for model_name in self.configs["models_to_run"].keys():
            model = self.model_class_names[
                model_name
            ]  # You can use any scikit-learn model

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
                    model = self.model_class_names[model_name](**params_dict)
                    # todo : add cross validation
                    model.fit(X_train, y_train)

                    # Log the hyperparameters
                    mlflow.log_params(params_dict)
                    mlflow.log_param("model_name", model_name)
                    # Log evaluation metrics (e.g., accuracy, F1-score)
                    accuracy = model.score(X_test, y_test)
                    mlflow.log_metric("accuracy", accuracy)

                    # Log the trained model as an artifact
                    mlflow.sklearn.log_model(model, "model")
                    # todo : add model signature
                    # todo : add all test metrics
                    # todo : add feature of regression
