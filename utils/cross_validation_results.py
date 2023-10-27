import pandas as pd
from sklearn.model_selection import KFold

from utils.evaluate_model_prediction import (
    evaluate_classification_model_prediction,
    evaluate_regression_model_prediction,
)


def get_cross_validation_results(model, configs, X_train, y_train):
    #  Cross-Validation Setup
    kf = KFold(
        n_splits=configs["num_folds"],
        shuffle=True,
        random_state=42,
    )

    # Initialize a dataframe to store cross-validation scores
    cv_scorees_df = pd.DataFrame()
    # i will used to make index like fold0,fold1....
    i = 0
    # Perform cross-validation
    for train_index, val_index in kf.split(X_train):
        X_train_, X_val = X_train[train_index], X_train[val_index]
        y_train_, y_val = y_train[train_index], y_train[val_index]

        # Fit the model on the training data
        model.fit(X_train_, y_train_)

        # Evaluate the model on the validation data
        val_prediction = model.predict(X_val)
        if configs["model_type"] == "classification":
            eval_func = eval("evaluate_classification_model_prediction")
        if configs["model_type"] == "regression":
            eval_func = eval("evaluate_regression_model_prediction")
        val_evaluation_dict = eval_func(y_val, val_prediction, "val")
        cv_scorees_df = pd.concat(
            [
                cv_scorees_df,
                pd.DataFrame(val_evaluation_dict, index=[f"fold{i+1}"]),
            ]
        )

        i += 1

    # Display results
    # print(f"Cross-validation scores: {cv_scores}")
    # print(f"Mean CV accuracy: {sum(cv_scores) / len(cv_scores)}")

    # todo: take the path from config file
    cv_scorees_df.to_csv(
        configs["artifact_path"] + "cross_validation_scores.csv",
        index=True,
    )
