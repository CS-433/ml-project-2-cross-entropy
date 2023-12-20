import numpy as np
from sklearn.linear_model import LassoCV

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class LassoMethod(BaseMethod):
    """
    A concrete implementation of the BaseMethod for Lasso Regression with built-in cross-validation.

    This class extends the BaseMethod and implements the required `train` and `predict`
    methods using the LassoCV model from scikit-learn, which applies Lasso (L1) regularization
    with automatic tuning of the regularization parameter.

    Attributes:
        lasso (LassoCV): An instance of the LassoCV model from scikit-learn.

    Methods:
        train: Implements the training procedure for the Lasso Regression model.
        predict: Implements the prediction process using the trained Lasso Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initializes the LassoMethod instance.

        Args:
            dataloader (DataLoader): A data loader object which is expected to
                                     provide the necessary data for the method.

        This method initializes the base class and sets up the LassoCV model.
        """
        super().__init__(dataloader)
        self.lasso = LassoCV()

    def train(self) -> None:
        """
        Trains the Lasso Regression model with cross-validation.

        This method fits the LassoCV model to the data provided by the
        dataloader. It assumes that the dataloader has attributes `X` for the
        input features and `y` for the target variable.
        """
        self.lasso.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained Lasso Regression model.

        Args:
            x (np.ndarray): The input data for which predictions are to be made.

        Returns:
            np.ndarray: The predicted values as a NumPy array. The output format
                        is dictated by scikit-learn's LassoCV model.
        """
        return self.lasso.predict(x)
