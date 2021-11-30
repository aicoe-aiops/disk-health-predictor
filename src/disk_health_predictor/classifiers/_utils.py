import os
from typing import Dict

from ._prophetstor import PSDiskHealthClassifier
from ._redhat import RHDiskHealthClassifier


def get_modelstore_path() -> str:
    path = os.path.abspath(__file__)
    modelstore_path = os.path.join(os.path.dirname(path), "../pretrained_models")
    return modelstore_path


def get_optimal_classifier_name(config: Dict):
    return "redhat"


def DiskHealthClassifierFactory(predictor_name: str):
    if predictor_name == "redhat":
        predictor = RHDiskHealthClassifier()
        predictor.initialize(os.path.join(get_modelstore_path(), "redhat"))
    elif predictor_name == "prophetstor":
        predictor = PSDiskHealthClassifier()
        predictor.initialize(os.path.join(get_modelstore_path(), "prophetstor"))
    else:
        raise ValueError(f"No pretrained model found with the name {predictor_name}")
    return predictor
