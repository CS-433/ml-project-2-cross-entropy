from .base_method import BaseMethod


class MethodList(BaseMethod):
    def __init__(self, methods: list):
        self.methods = methods

    def fit(self, x, y=None):
        for method in self.methods:
            method.fit(x, y)

    def preprocess(self, x, y=None):
        for method in self.methods:
            x = method.preprocess(x, y)
        return x

    def fit_preprocess(self, x, y=None):
        for method in self.methods:
            x = method.fit_preprocess(x, y)
        return x

    def backward(self, x, y=None):
        for method in reversed(self.methods):
            x = method.backward(x, y)
        return x
