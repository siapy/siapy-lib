import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from segmentator.base_decision_algo import BaseDecisionAlgo


class Lda(BaseDecisionAlgo):
    def __init__(self, cfg):
        self.model = LinearDiscriminantAnalysis()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict([X])


class Svm(BaseDecisionAlgo):
    def __init__(self, cfg):
        self.model = SVC()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict([X])

class Sid(BaseDecisionAlgo):
    def __init__(self, cfg):
        self.cls_remove = cfg.misc.segmentator.classes_remove
        self.cls_keep = cfg.misc.segmentator.classes_keep
        if not(len(self.cls_keep) == len(self.cls_remove) == 1):
            raise ValueError("Sid only works with one class removed and one class kept")
        self.cls_keep = self.cls_keep[0]
        self.cls_remove = self.cls_remove[0]

        self.background_sig = None
        self.sid_threshold = None

    @staticmethod
    def sid(s1, s2):
        p = s1 + np.spacing(1)
        q = s2 + np.spacing(1)
        return np.sum(p * np.log(p / q) + q * np.log(q / p))

    def _fit(self, X, y):
        y_inv = self.encoder.inverse_transform(y)
        X_remove = X[y_inv == self.cls_remove]
        X_keep = X[y_inv == self.cls_keep]

        self.background_sig = np.mean(X_remove, axis=0)
        # calculate sid for all extracted signatures
        sid_remove = list(map(lambda sig: Sid.sid(sig, self.background_sig), X_remove))
        sid_keep = list(map(lambda sig: Sid.sid(sig, self.background_sig), X_keep))

        # make column vecotrs
        sid_keep = np.atleast_2d(sid_keep).T
        sid_remove = np.atleast_2d(sid_remove).T

        # here svc is used to determine the threshold
        X = np.concatenate((sid_keep, sid_remove), axis=0)
        y = [0] * len(sid_keep) + [1] * len(sid_remove)
        clf = SVC(kernel='linear').fit(X, y)
        # calculate boundary by equation:
        # w0*x + b = 0 -> x = -b/w0
        self.sid_threshold = - clf.intercept_/clf.coef_

    def _predict(self, X):
        sid = Sid.sid(self.background_sig, X)
        if sid > self.sid_threshold:
            return self.cls_keep
        else:
            return self.cls_remove
