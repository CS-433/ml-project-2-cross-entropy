from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from methods.base_method import BaseMethod


class PCAAnalysis(BaseMethod):
    def __init__(self, dataloader, n_components=None, standardize=False):
        """
        Args:
            dataloader (Dataloader): 
            n_components (int, optional): Numbers of components to keep. 
                                          Defaults to None (i.e (num_features - 1) in sklearn).
        """
        super().__init__(dataloader)
        self.n_components = n_components
        self.standardize = standardize
        self.model = PCA(n_components=self.n_components)
        self.scaler = StandardScaler() if standardize else None

    def _standardize_data(self, X):
        if self.standardize:
            return self.scaler.fit_transform(X)
        return X

    def train(self):
        X = self._standardize_data(self.dataloader.X)
        return self.model.fit(X)

    def transform(self, x):
        x = self._standardize_data(x)
        return self.model.transform(x)

    def fit_transform(self):
        X = self._standardize_data(self.dataloader.X)
        return self.model.fit_transform(X)

    def get_explained_variance_ratio(self):
        if hasattr(self.model, "explained_variance_ratio_"):
            return self.model.explained_variance_ratio_
        else:
            raise RuntimeError(
                "PCA model must be trained before getting explained variance ratio."
            )

    def find_components_number(self, ratio=1 - 10e-6):
        if not hasattr(self.model, "explained_variance_ratio_"):
            raise RuntimeError(
                "PCA model must be trained before searching for the number of components."
            )

        cumulative_variance = self.model.explained_variance_ratio_.cumsum()

        n_components = sum(cumulative_variance < ratio) + 1

        return n_components
