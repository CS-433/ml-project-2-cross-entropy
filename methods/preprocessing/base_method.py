import abc

class PreprocessingMethod(abc.ABC):
    @abc.abstractmethod
    def fit(self, x, y=None):
        pass

    @abc.abstractmethod
    def preprocess(self, x, y=None):
        pass

    @abc.abstractmethod
    def fit_preprocess(self, x, y=None):
        pass

    @abc.abstractmethod
    def backward(self, x, y=None):
        pass
