from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from models.classification.classification_proecessor import ClassificationModels
from models.regression.regression_processors import RegressionModels

model_class_names = {"random_forest": RandomForestClassifier, "svm": svm.SVC}

model_types = {"regression": RegressionModels, "classification": ClassificationModels}
