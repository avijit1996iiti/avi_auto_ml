import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from itertools import product
import mlflow
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold, train_test_split

from utils.evaluate_model_prediction import evaluate_classification_model_prediction
from utils.cross_validation_results import get_cross_validation_results
from utils.log_model_with_signature import log_model_with_signature


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

        # todo: include stratfied sampling
        # Split the Data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.configs["train_test_split"]["test_size"],
            random_state=42,
        )

        for model_name in self.configs["models_to_run"].keys():
            # Define Hyperparameter Grid
            param_grid = self.configs["models_to_run"][model_name]

            # Generate all possible combinations of hyperparameters
            param_combinations = list(product(*param_grid.values()))

            # perform grid search with logging every information in mlflow
            for params_dict in pd.DataFrame(
                param_combinations, columns=param_grid.keys()
            ).to_dict("r"):
                with mlflow.start_run():
                    # Create and train the model with the current hyperparameters
                    # model = self.model_class_names[model_name](**params_dict)
                    model = self.model_class_names[self.configs["model_type"]][
                        model_name
                    ](**params_dict)
                    get_cross_validation_results(model, self.configs, X_train, y_train)
                    mlflow.log_artifact(
                        self.configs["artifact_path"] + "cross_validation_scores.csv"
                    )
                    # Assessing Model Performance
                    # todo: add mean and std for each metric
                    """
                    mean_accuracy = np.mean(cv_scores)
                    std_accuracy = np.std(cv_scores)
                    mlflow.log_metrics(
                        {
                            "cv_mean_accuracy": mean_accuracy,
                            "cv_std_accuracy": std_accuracy,
                        }
                    )
                    """

                    # Log the hyperparameters
                    mlflow.log_params(params_dict)
                    mlflow.log_param("model_name", model_name)

                    # todo : add cross validation
                    model.fit(X_train, y_train)

                    train_prediction = model.predict(X_train)
                    train_evaluation_dict = evaluate_classification_model_prediction(
                        y_train, train_prediction, "train"
                    )

                    mlflow.log_metrics(train_evaluation_dict)

                    # Log evaluation metrics
                    test_prediction = model.predict(X_test)
                    test_evaluation_dict = evaluate_classification_model_prediction(
                        y_test, test_prediction, "test"
                    )

                    mlflow.log_metrics(test_evaluation_dict)
                    log_model_with_signature(
                        model=model,
                        model_data_df=model_data_df,
                        configs=self.configs,
                        train_prediction=train_prediction,
                    )

                    # todo : add all test metrics
                    # todo : add cross validation feature
