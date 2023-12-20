from .base_method import PreprocessingMethod


class ShiftingMethod(PreprocessingMethod):
    def __init__(self, shift: float):
        self.shift = shift

    def fit(self, x, y=None):
        return self

    def preprocess(self, x, y=None):
        return x - self.shift

    def backward(self, x, y=None):
        return x + self.shift

    def fit_preprocess(self, x, y=None):
        return self.preprocess(x)
