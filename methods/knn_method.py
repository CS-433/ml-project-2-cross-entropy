from sklearn.neighbors import KNeighborsRegressor

from methods.base_method import BaseMethod


class KnnMethod(BaseMethod):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.knn = KNeighborsRegressor()

    def train(self):
        self.knn.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.knn.predict(x)