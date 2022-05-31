from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from segmentator.base_decision_algo import BaseDecisionAlgo


class lda(BaseDecisionAlgo):
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict([X])


class svm(BaseDecisionAlgo):
    def __init__(self):
        self.model = SVC()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict([X])

# class sid(BaseDecisionAlgo):

