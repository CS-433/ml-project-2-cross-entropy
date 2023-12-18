from sklearn.linear_model import LassoLarsCV

from methods.base_method import BaseMethod


class LassoLarsMethod(BaseMethod):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.lasso_lars = LassoLarsCV()

    def train(self):
        self.lasso_lars.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.lasso_lars.predict(x)