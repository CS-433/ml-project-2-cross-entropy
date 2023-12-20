import numpy as np
from sklearn.linear_model import BayesianRidge

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class BayesianRidgeMethod(BaseMethod):
    """
        A concrete implementation of the BaseMethod for Bayesian Ridge Regression.

        This class extends the BaseMethod and implements the required `train` and
        `predict` methods using the BayesianRidge model from scikit-learn.

        Attributes:
            bayesian_ridge (BayesianRidge): An instance of the BayesianRidge model
                                            from scikit-learn.

        Methods:
            train: Implements the training procedure for the Bayesian Ridge Regression model.
            predict: Implements the prediction process using the trained Bayesian Ridge Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
            Initializes the BayesianRidgeMethod instance.

            Args:
                dataloader (Any): A data loader object which is expected to
                                provide the necessary data for the method.

            This method initializes the base class and sets up the BayesianRidge model.
        """
        super().__init__(dataloader)
        self.bayesian_ridge = BayesianRidge()

    def train(self) -> None:
        """
            Trains the Bayesian Ridge Regression model.

            This method fits the BayesianRidge model to the data provided by the
            dataloader. It assumes that the dataloader has attributes `X` for the
            input features and `y` for the target variable.
        """
        self.bayesian_ridge.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
            Makes predictions using the trained Bayesian Ridge Regression model.

            Args:
                x (np.ndarray): The input data for which predictions are to be made.

            Returns:
                np.ndarray: The predicted values. The format and type of the output
                    depend on the implementation in scikit-learn's BayesianRidge model.
        """
        return self.bayesian_ridge.predict(x)
