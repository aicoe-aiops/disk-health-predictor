from typing import Dict

from ._prophetstor import PSDiskHealthClassifier
from ._redhat import RHDiskHealthClassifier


def get_optimal_classifier_name(config: Dict):
    return "redhat"


def DiskHealthClassifierFactory(predictor_name: str):
    if predictor_name == "redhat":
        return RHDiskHealthClassifier
    elif predictor_name == "prophetstor":
        return PSDiskHealthClassifier
    else:
        raise ValueError(f"No pretrained model found with the name {predictor_name}")
