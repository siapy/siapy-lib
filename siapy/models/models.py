from typing import Protocol


class ClassificationAlgorithm(Protocol):
    def fit(self, X, y): ...

    def predict(self, X): ...


class ClassificationModel:
    def __init__(self, model: ClassificationAlgorithm):
        self._model = model

    @property
    def model(self) -> ClassificationAlgorithm:
        return self._model
