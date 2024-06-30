import numpy as np
from numba import njit, prange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from siapy.segmentator import BaseDecisionAlgo


class KeepAll(BaseDecisionAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cls_keep_enc = None

    def _fit(self, X, y):
        self.cls_keep_enc = self.encoder.transform(self.cfg.segmentator.classes_keep)[0]

    def _predict(self, X):
        return [self.cls_keep_enc] * len(X)


class Lda(BaseDecisionAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = LinearDiscriminantAnalysis()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)


class Svm(BaseDecisionAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = SVC()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)


class Sid(BaseDecisionAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cls_remove = cfg.segmentator.classes_remove
        self.cls_keep = cfg.segmentator.classes_keep
        if not (len(self.cls_keep) == len(self.cls_remove) == 1):
            raise ValueError("Sid only works with one class removed and one class kept")
        self.cls_keep = self.cls_keep[0]
        self.cls_remove = self.cls_remove[0]

        self.cls_keep_enc = None
        self.cls_remove_enc = None

        self.background_sig = None
        self.sid_threshold = None

    def _fit(self, X, y):
        y_inv = self.encoder.inverse_transform(y)
        X_remove = X[y_inv == self.cls_remove]
        X_keep = X[y_inv == self.cls_keep]

        self.background_sig = np.mean(X_remove, axis=0)
        # calculate sid for all extracted signatures
        sid_remove = list(map(lambda sig: self.sid(sig, self.background_sig), X_remove))
        sid_keep = list(map(lambda sig: self.sid(sig, self.background_sig), X_keep))

        # make column vecotrs
        sid_keep = np.atleast_2d(sid_keep).T
        sid_remove = np.atleast_2d(sid_remove).T

        # here svc is used to determine the threshold
        X = np.concatenate((sid_keep, sid_remove), axis=0)
        y = [0] * len(sid_keep) + [1] * len(sid_remove)
        clf = SVC(kernel="linear").fit(X, y)
        # calculate boundary by equation:
        # w0*x + b = 0 -> x = -b/w0
        self.sid_threshold = -clf.intercept_ / clf.coef_

    def _predict(self, X):
        self.cls_keep_enc = self.encoder.transform([self.cls_keep])[0]
        self.cls_remove_enc = self.encoder.transform([self.cls_remove])[0]
        targets = np.array(self.sid_multi(np.array(X), self.background_sig, self.sid))
        # replace values above threshold with class keep and below with class remove
        targets = np.where(
            targets > self.sid_threshold, self.cls_keep_enc, self.cls_remove_enc
        )[0]
        return targets

    @staticmethod
    @njit()
    def sid(p, q):
        p = p + np.spacing(1)
        q = q + np.spacing(1)
        return np.sum(p * np.log(p / q) + q * np.log(q / p))

    @staticmethod
    @njit(parallel=True)
    def sid_multi(X, background_sig, sid_func):
        n_rows, _ = X.shape
        sid_values = np.zeros(n_rows)
        for idx in prange(n_rows):
            sid_values[idx] = sid_func(X[idx, :], background_sig)
        return sid_values
