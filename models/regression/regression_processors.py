from itertools import product
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split

from utils.evaluate_model_prediction import evaluate_regression_model_prediction
from utils.cross_validation_results import get_cross_validation_results
from utils.log_model_with_signature import log_model_with_signature


class RegressionModels:
    def __init__(self, model_class_names, configs) -> None:
        self.model_class_names = model_class_names
        self.configs = configs

        self._execute()

    def _execute(self):
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
                    model = self.model_class_names[self.configs["model_type"]][
                        model_name
                    ](
                        **params_dict
                    )  # You can use any scikit-learn model
                    # todo : add cross validation
                    get_cross_validation_results(model, self.configs, X_train, y_train)

                    mlflow.log_artifact(
                        self.configs["artifact_path"] + "cross_validation_scores.csv",
                    )
                    # Assessing Model Performance
                    """
                    mean_rmse = np.mean(rmse_scores)
                    std_rmse = np.std(rmse_scores)
                    mlflow.log_metrics(
                        {"cv_mean_rmse": mean_rmse, "cv_std_rmse": std_rmse}
                    )
                    """
                    # Log the hyperparameters
                    mlflow.log_params(params_dict)
                    mlflow.log_param("model_name", model_name)
                    model.fit(X_train, y_train)
                    train_prediction = model.predict(X_train)
                    train_evaluation_dict = evaluate_regression_model_prediction(
                        y_train, train_prediction, "train"
                    )

                    mlflow.log_metrics(train_evaluation_dict)

                    # Evaluate the model on the test set
                    test_prediction = model.predict(X_test)
                    test_evaluation_dict = evaluate_regression_model_prediction(
                        y_test, test_prediction, "test"
                    )
                    mlflow.log_metrics(test_evaluation_dict)

                    # Log the trained model as an artifact
                    log_model_with_signature(
                        model=model,
                        model_data_df=model_data_df,
                        configs=self.configs,
                        train_prediction=train_prediction,
                    )
                    # mlflow.sklearn.log_model(model, "model")
                    # todo : add model signature
                    # todo : explore different options to get cv metrics
