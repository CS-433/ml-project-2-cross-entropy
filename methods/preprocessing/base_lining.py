import numpy as np
from sklearn.linear_model import RidgeCV

from .base_method import PreprocessingMethod


class BaselingMethod(PreprocessingMethod):
    def __init__(self,
                 alphas: np.ndarray = np.geomspace(1e-18, 1e2, 14),
                 fit_intercept: bool = False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.clf = RidgeCV(alphas=alphas, fit_intercept=fit_intercept)

    def fit(self, x, y):
        self.clf.fit(x, y)
        return self

    def preprocess(self, x, y):
        return y - self.clf.predict(x)

    def backward(self, x, y):
        return y + self.clf.predict(x)

    def fit_preprocess(self, x, y):
        self.fit(x, y)
        return self.preprocess(x, y)
