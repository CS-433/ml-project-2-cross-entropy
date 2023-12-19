from sklearn.preprocessing import StandardScaler
from .base_method import PreprocessingMethod

class StandardizationMethod(PreprocessingMethod):

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, x, y=None):
        self.scaler.fit(x)

    def preprocess(self, x, y=None):
        return self.scaler.transform(x)

    def backward(self, x, y=None):
        return self.scaler.inverse_transform(x)

    def fit_preprocess(self, x, y=None):
        self.fit(x)
        return self.preprocess(x)
