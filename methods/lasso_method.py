from sklearn.linear_model import LassoCV

from methods.base_method import BaseMethod


class LassoMethod(BaseMethod):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.lasso = LassoCV()

    def train(self):
        self.lasso.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.lasso.predict(x)
