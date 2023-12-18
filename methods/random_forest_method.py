from sklearn.ensemble import RandomForestRegressor

from methods.base_method import BaseMethod


class RandomForestMethod(BaseMethod):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.random_forest = RandomForestRegressor()

    def train(self):
        self.random_forest.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.random_forest.predict(x)
