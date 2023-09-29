from utils.configs import read_yaml_file
from utils.model_types import model_class_names

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from itertools import product
import mlflow
import pandas as pd


def ab_auto_runner(configs_path: str):
    # read configurations from the specified yaml file
    configs = read_yaml_file(configs_path)
    # todo : read model data and change below line
    data = load_iris()
    X, y = data.data, data.target

    # Split the Data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=configs["train_test_split"]["test_size"], random_state=42
    )

    for model_name in configs["models_to_run"].keys():
        print(
            model_name,
            model_class_names[model_name],
            configs["train_test_split"]["test_size"],
        )
        model = model_class_names[model_name]  # You can use any scikit-learn model

        # Step 5: Define Hyperparameter Grid
        param_grid = configs["models_to_run"][model_name]

        # Step 6: Generate all possible combinations of hyperparameters
        param_combinations = list(product(*param_grid.values()))

        # perform grid search with logging every information in mlflow
        for params_dict in pd.DataFrame(
            param_combinations, columns=param_grid.keys()
        ).to_dict("r"):
            with mlflow.start_run():
                # Create and train the model with the current hyperparameters
                model = model_class_names[model_name](**params_dict)
                model.fit(X_train, y_train)

                # Log the hyperparameters
                mlflow.log_params(params_dict)

                # Log evaluation metrics (e.g., accuracy, F1-score)
                accuracy = model.score(X_test, y_test)
                mlflow.log_metric("accuracy", accuracy)

                # Log the trained model as an artifact
                mlflow.sklearn.log_model(model, "model")
                # todo : add model signature
                # todo : add all test metrics
                # todo : add feature of regression
