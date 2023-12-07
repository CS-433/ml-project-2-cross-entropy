import numpy as np
from sklearn.linear_model import RidgeCV

from methods.base_method import BaseMethod


class RidgeMethod(BaseMethod):
    def __init__(self):
        super(RidgeMethod).__init__()
        self.clf = RidgeCV(alphas=np.geomspace(1e-12, 1e2, 32), fit_intercept=True)


    def train(self):
        self.clf.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x):
        return self.clf.predict(x)
