import numpy as np
from sklearn.linear_model import LassoLarsCV

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class LassoLarsMethod(BaseMethod):
    """
    A concrete implementation of the BaseMethod for Lasso Regression using
    the Least Angle Regression (LARS) algorithm with built-in cross-validation.

    This class extends BaseMethod and implements the required `train` and `predict`
    methods using the LassoLarsCV model from scikit-learn.

    Attributes:
        lasso_lars (LassoLarsCV): An instance of the LassoLarsCV model from scikit-learn.

    Methods:
        train: Implements the training procedure for the Lasso LARS Regression model.
        predict: Implements the prediction process using the trained Lasso LARS Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initializes the LassoLarsMethod instance.

        Args:
            dataloader (DataLoader): A data loader object which is expected to
                                     provide the necessary data for the method.

        This method initializes the base class and sets up the LassoLarsCV model.
        """
        super().__init__(dataloader)
        self.lasso_lars = LassoLarsCV()

    def train(self) -> None:
        """
        Trains the Lasso Regression model using the LARS algorithm with cross-validation.

        This method fits the LassoLarsCV model to the data provided by the
        dataloader. It assumes that the dataloader has attributes `X` for the
        input features and `y` for the target variable.
        """
        self.lasso_lars.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained Lasso LARS Regression model.

        Args:
            x (np.ndarray): The input data for which predictions are to be made.

        Returns:
            np.ndarray: The predicted values as a NumPy array. The output format
                        is dictated by scikit-learn's LassoLarsCV model.
        """
        return self.lasso_lars.predict(x)
