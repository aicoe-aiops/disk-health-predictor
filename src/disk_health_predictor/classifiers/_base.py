"""Abstract base class for all classifiers. Defines the API for classifiers."""
from abc import abstractmethod


class DiskHealthClassifier:
    """
    Base class for classifiers (good/bad/warning)
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize(self, model_dir: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, dataset) -> str:
        raise NotImplementedError()
