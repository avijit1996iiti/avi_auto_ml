import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature


def log_model_with_signature(
    model, model_data_df: pd.DataFrame, configs: dict, train_prediction
):
    # Log the trained model as an artifact
    signature = infer_signature(
        model_data_df[configs["independent_variables"]],
        train_prediction,
    )

    # Log the model with the specified signature
    mlflow.sklearn.log_model(model, "model", signature=signature)
