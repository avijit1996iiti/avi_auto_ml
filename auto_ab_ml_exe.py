from models.model_processor import auto_ab_runner

auto_ab_runner(configs_path="config_classification.yaml")
# auto_ab_runner(configs_path="config_regression.yaml")
print("Execution Done")
