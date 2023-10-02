## AutoABML

## Overview
AutoABML stands as a cutting-edge Python library engineered to transform the machine learning sphere through the automation of the entire model selection and hyperparameter tuning process. Its primary goal is to provide a seamless experience for both novices and seasoned data scientists, enabling them to effortlessly craft high-performance machine learning models. The library not only simplifies the complex tasks of model selection and hyperparameter tuning but also empowers users to navigate the complete lifecycle of their ML models.

A standout feature of AutoABML is its integration with MLflow, a robust platform for managing the end-to-end machine learning lifecycle. By leveraging MLflow, the library enables users to track and monitor experiments effectively. This functionality proves invaluable for maintaining a comprehensive record of model development, experimentation, and performance metrics.

Whether you're just starting your journey in machine learning or you're an experienced practitioner, AutoABML provides a user-friendly interface that abstracts away the complexities of the model-building process. The library's automation capabilities not only save time but also enhance the likelihood of discovering optimal model configurations. With AutoABML, the aim is to democratize machine learning by making it accessible to a broader audience while ensuring the creation of highly performant models.
## Installation
Package not on pypy yet. Clone the repo, run auto_ab_ml_exe.py with your yaml file path.
## Getting Started

\# import required modules

from models.model_processor import auto_ab_runner

\# provide the path of your configuration file 

\# auto_ab_runner(configs_path="config_classification.yaml")

auto_ab_runner(configs_path="config_regression.yaml")

print("Execution Done")

## Example Configurations

### Classification

```
model_type: "classification" # specify the model type 

model_data: "data/input/classification/classification_dummy_data.csv" # provide data path required for training and testing

independent_variables: ['0','1'] # specify the list of independent_variables

dependent_variable: 'target' # specify the dependent_variable

number_of_classes: 3 # specify number of 

# dictionary containing model names to run along with theier hyperparameter grid
# currently random_forest and svm are supported
models_to_run: {
  'random_forest': {
    
    'n_estimators': [50, 100, 150],
    
    'max_depth': [ 10, 20, 30], # None is not supported as of now
    
    \#'min_samples_split': [2, 5, 10],
    
    \#'min_samples_leaf': [1, 2, 4],
    
    \#'max_features': ['auto', 'sqrt', 'log2'],
    
    \#'bootstrap': [True, False]

},

'svm': {

  'C': [1,2,3, 0.75],

  'kernel': ['linear', 'poly', 'rbf', 'sigmoid',] # 'precomputed'
}

}

# specify test_size as the fraction of complete data 
train_test_split: { 
  
  test_size: 0.2 

} 

```


### Regression 

``` 

model_type: "regression" # specify the model type 

model_data: "data/input/regression/regression_dummy_data.csv" # provide data path required for training and testing

independent_variables: ['0','1','2','3','4','5','6','7','8','9','10','11','12'] # provide data path required for training and testing

dependent_variable: 'target' # specify the dependent_variable


# dictionary containing model names to run along with theier hyperparameter grid
# currently random_forest_regressor and support_vector_regressor are supported
models_to_run: {

  "random_forest_regressor": {

    'n_estimators': [50, 100, 150],
    
    'max_depth': [ 10, 20, 30], # None
    
    #'min_samples_split': [2, 5, 10],
    
    #'min_samples_leaf': [1, 2, 4],
    
    #'max_features': ['auto', 'sqrt', 'log2'],
    
    #'bootstrap': [True, False]
},
"support_vector_regressor": {
  'C': [0.1, 1, 10],
  
  'epsilon': [0.01, 0.1, 0.2],
  
  'kernel': ['linear', 'rbf', 'poly']
}

}

# specify test_size as the fraction of complete data 
train_test_split: { 
  
  test_size: 0.2 


} 
```
## Acknowledgements

 - [Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](https://scikit-learn.org/stable/about.html)
 - [mlflow](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
 

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Authors

- [@avijit1996iiti](https://github.com/avijit1996iiti)

