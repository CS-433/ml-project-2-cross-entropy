import numpy as np
from sklearn.linear_model import RidgeCV

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class RidgeMethod(BaseMethod):
    """
    A concrete implementation of the BaseMethod for Ridge Regression with built-in cross-validation.

    This class extends the BaseMethod and implements the required `train` and `predict`
    methods using the RidgeCV model from scikit-learn. RidgeCV applies Ridge (L2) regularization
    and automatically tunes the regularization parameter (alpha) using cross-validation.

    Attributes:
        clf (RidgeCV): An instance of the RidgeCV model from scikit-learn with specified alpha values
                       and fit_intercept set to True.

    Methods:
        train: Implements the training procedure for the Ridge Regression model.
        predict: Implements the prediction process using the trained Ridge Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initializes the RidgeMethod instance.

        Args:
            dataloader (DataLoader): A data loader object which is expected to
                                     provide the necessary data for the method.

        This method initializes the base class and sets up the RidgeCV model with a geometric
        sequence of alpha values ranging from 1e-12 to 1e2, and with fit_intercept set to True.
        """
        super().__init__(dataloader)
        self.clf = RidgeCV(alphas=np.geomspace(1e-12, 1e2, 32), fit_intercept=True)

    def train(self) -> None:
        """
        Trains the Ridge Regression model with cross-validation.

        This method fits the RidgeCV model to the data provided by the
        dataloader. It assumes that the dataloader has attributes `X` for the
        input features and `y` for the target variable.
        """
        self.clf.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained Ridge Regression model.

        Args:
            x (np.ndarray): The input data for which predictions are to be made.

        Returns:
            np.ndarray: The predicted values as a NumPy array. The output format
                        is dictated by scikit-learn's RidgeCV model.
        """
        return self.clf.predict(x)
