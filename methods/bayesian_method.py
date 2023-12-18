from sklearn.linear_model import BayesianRidge

from methods.base_method import BaseMethod


class BayesianRidgeMethod(BaseMethod):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.bayesian_ridge = BayesianRidge()

    def train(self):
        self.bayesian_ridge.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.bayesian_ridge.predict(x)