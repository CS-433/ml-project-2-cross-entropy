from .base_method import PreprocessingMethod

class IdentityMethod(PreprocessingMethod):
    """Simply returns the input."""
    def fit(self, x, y=None):
        return self

    def preprocess(self, x, y=None):
        return x

    def backward(self, x, y=None):
        return x

    def fit_preprocess(self, x, y=None):
        return self.preprocess(x)
