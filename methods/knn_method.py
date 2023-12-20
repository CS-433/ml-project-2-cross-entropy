import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class KnnMethod(BaseMethod):
    """
    A concrete implementation of the BaseMethod for K-Nearest Neighbors (KNN) Regression.

    This class extends the BaseMethod and implements the required `train` and `predict`
    methods using the KNeighborsRegressor model from scikit-learn.

    Attributes:
        knn (KNeighborsRegressor): An instance of the KNeighborsRegressor model from scikit-learn.

    Methods:
        train: Implements the training procedure for the KNN Regression model.
        predict: Implements the prediction process using the trained KNN Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initializes the KnnMethod instance.

        Args:
            dataloader (DataLoader): A data loader object which is expected to
                                     provide the necessary data for the method.

        This method initializes the base class and sets up the KNeighborsRegressor model.
        """
        super().__init__(dataloader)
        self.knn = KNeighborsRegressor()

    def train(self) -> None:
        """
        Trains the K-Nearest Neighbors Regression model.

        This method fits the KNeighborsRegressor model to the data provided by the
        dataloader. It assumes that the dataloader has attributes `X` for the
        input features and `y` for the target variable.
        """
        self.knn.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained K-Nearest Neighbors Regression model.

        Args:
            x (np.ndarray): The input data for which predictions are to be made.

        Returns:
            np.ndarray: The predicted values as a NumPy array. The output format
                        is dictated by scikit-learn's KNeighborsRegressor model.
        """
        return self.knn.predict(x)
