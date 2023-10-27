def evaluate_classification_model_prediction(y_true, y_pred, segement_of_data):
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


def evaluate_regression_model_prediction(y_true, y_pred, segement_of_data):
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import numpy as np

    # Measures the average absolute differences between actual and predicted values.
    mae = mean_absolute_error(y_true, y_pred)
    # Measures the average squared differences between actual and predicted values.
    mse = mean_squared_error(y_true, y_pred)
    # Represents the square root of the MSE, providing an interpretable scale.
    rmse = np.sqrt(mse)
    # Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
    r2 = r2_score(y_true, y_pred)
    # Measures the proportion to which the model explains the variance of the target variable.
    explained_variance = explained_variance_score(y_true, y_pred)

    return {
        f"{segement_of_data}_mae": mae,
        f"{segement_of_data}_mse": mse,
        f"{segement_of_data}_rmse": rmse,
        f"{segement_of_data}_r2": r2,
        f"{segement_of_data}_explained_variance": explained_variance,
    }
