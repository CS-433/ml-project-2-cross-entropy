import numpy as np
from sklearn.preprocessing import Normalizer
from .base_method import PreprocessingMethod

class NormalizationMethod(PreprocessingMethod):
    def __init__(self):
        self.scaler = Normalizer()

    def fit(self, x, y=None):
        self.max = np.max(x)
        self.min = np.min(x)
        self.scaler.fit(x)

    def preprocess(self, x, y=None):
        return self.scaler.transform(x)

    def backward(self, x, y=None):
        x = x * (self.max - self.min) + self.min
        return x

    def fit_preprocess(self, x, y=None):
        self.fit(x)
        return self.preprocess(x)
