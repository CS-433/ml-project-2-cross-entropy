from sklearn.decomposition import PCA
from .base_method import PreprocessingMethod

class PCAMethod(PreprocessingMethod):
    def __init__(self, n_components=8):
        self.pca = PCA(n_components=n_components)

    def fit(self, x, y=None):
        return self.pca.fit(x)

    def preprocess(self, x, y=None):
        return self.pca.transform(x)

    def fit_preprocess(self, x, y=None):
        self.fit(x)
        return self.preprocess(x)

    def backward(self, x, y=None):
        raise RuntimeError("PCA method does not support backpropagation.")
