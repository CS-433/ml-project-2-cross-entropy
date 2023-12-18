from sklearn.tree import DecisionTreeRegressor

from methods.base_method import BaseMethod


class DecisionTreeMethod(BaseMethod):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.decision_tree = DecisionTreeRegressor()

    def train(self):
        self.decision_tree.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.decision_tree.predict(x)