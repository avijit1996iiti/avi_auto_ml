from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from models.classification.classification_proecessor import ClassificationModels
from models.regression.regression_processors import RegressionModels
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

model_class_names = {
    "classification": {"random_forest": RandomForestClassifier, "svm": svm.SVC},
    "regression": {
        "random_forest_regressor": RandomForestRegressor,
        "support_vector_regressor": SVR,
    },
}
model_types = {"regression": RegressionModels, "classification": ClassificationModels}
