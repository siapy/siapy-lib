from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from segmentator.base_decision_algo import BaseDecisionAlgo


class lda(BaseDecisionAlgo):
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)


