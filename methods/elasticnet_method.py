from sklearn.linear_model import ElasticNetCV

from methods.base_method import BaseMethod


class ElasticNetMethod(BaseMethod):
    def __init__(self, dataloader):
        super().__init__(dataloader)
        self.elastic = ElasticNetCV()

    def train(self):
        self.elastic.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.elastic.predict(x)