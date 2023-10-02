from utils.configs import read_yaml_file
from utils.model_types import model_types, model_class_names


def auto_ab_runner(configs_path: str):
    # read configurations from the specified yaml file
    configs = read_yaml_file(configs_path)
    model_types[configs["model_type"]](model_class_names, configs)
