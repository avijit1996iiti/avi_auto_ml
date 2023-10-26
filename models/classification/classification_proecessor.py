import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from itertools import product
import mlflow
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold, train_test_split


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

                    #  Cross-Validation Setup
                    kf = KFold(
                        n_splits=self.configs["num_folds"],
                        shuffle=True,
                        random_state=42,
                    )

                    # Initialize a list to store cross-validation scores
                    cv_scores = []

                    # Perform cross-validation
                    for train_index, val_index in kf.split(X_train):
                        X_train_, X_val = X[train_index], X[val_index]
                        y_train_, y_val = y[train_index], y[val_index]

                        # Fit the model on the training data
                        model.fit(X_train_, y_train_)

                        # Evaluate the model on the validation data
                        val_score = model.score(X_val, y_val)
                        cv_scores.append(val_score)

                    # Display results
                    print(f"Cross-validation scores: {cv_scores}")
                    print(f"Mean CV accuracy: {sum(cv_scores) / len(cv_scores)}")

                    cv_scorees_df = pd.DataFrame(
                        columns=[f"fold{i+1}" for i in range(self.configs["num_folds"])]
                    )
                    cv_scorees_df.loc[0, :] = cv_scores
                    cv_scorees_df.to_csv(
                        "/home/avijit/selflearning/auto_ml/avi_auto_ml/data/output/classification/cross_validation_scores.csv",
                        index=None,
                    )
                    mlflow.log_artifact(
                        "/home/avijit/selflearning/auto_ml/avi_auto_ml/data/output/classification/cross_validation_scores.csv"
                    )
                    # Assessing Model Performance
                    mean_accuracy = np.mean(cv_scores)
                    std_accuracy = np.std(cv_scores)
                    mlflow.log_metrics(
                        {
                            "cv_mean_accuracy": mean_accuracy,
                            "cv_std_accuracy": std_accuracy,
                        }
                    )

                    # Log the hyperparameters
                    mlflow.log_params(params_dict)
                    mlflow.log_param("model_name", model_name)

                    # todo : add cross validation
                    model.fit(X_train, y_train)

                    train_prediction = model.predict(X_train)
                    train_evaluation_dict = self.evaluate_model_prediction(
                        y_train, train_prediction, "train"
                    )

                    mlflow.log_metrics(train_evaluation_dict)

                    # Log evaluation metrics
                    test_prediction = model.predict(X_test)
                    test_evaluation_dict = self.evaluate_model_prediction(
                        y_test, test_prediction, "test"
                    )

                    mlflow.log_metrics(test_evaluation_dict)

                    # Log the trained model as an artifact
                    mlflow.sklearn.log_model(model, "model")
                    # todo : add model signature
                    # todo : add all test metrics
                    # todo : add cross validation feature

    def evaluate_model_prediction(self, y_true, y_pred, segement_of_data):
        # Measures the proportion of correctly classified instances.
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import roc_auc_score

        accuracy = accuracy_score(y_true, y_pred)

        # Measures the accuracy of the positive predictions.
        precision = precision_score(y_true, y_pred, average="weighted")

        # Measures the ability of the model to capture all the positive instances.
        recall = recall_score(y_true, y_pred, average="weighted")

        # Represents the harmonic mean of precision and recall.
        f1 = f1_score(y_true, y_pred, average="weighted")

        # todo: include confusion matrix
        # A table showing the number of true positives, true negatives, false positives, and false negatives.
        # confusion_matrix_result = confusion_matrix(y_true, y_pred)
        # print(confusion_matrix_result)

        # ROC curve visualizes the performance of a binary classification model at various classification thresholds.
        # auc = roc_auc_score(y_true, y_pred)
        # todo: include roc_auc

        return {
            f"{segement_of_data}_accuracy": accuracy,
            f"{segement_of_data}_precision": precision,
            f"{segement_of_data}_recall": recall,
            f"{segement_of_data}_f1": f1,
        }
