import numpy as np
from sklearn.linear_model import ElasticNetCV

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class ElasticNetMethod(BaseMethod):
    """
    A concrete implementation of the BaseMethod for Elastic Net Regression with Cross-Validation.

    This class extends the BaseMethod and implements the required `train` and `predict`
    methods using the ElasticNetCV model from scikit-learn, which combines Elastic Net
    regression with built-in cross-validation.

    Attributes:
        elastic (ElasticNetCV): An instance of the ElasticNetCV model from scikit-learn.

    Methods:
        train: Implements the training procedure for the Elastic Net Regression model.
        predict: Implements the prediction process using the trained Elastic Net Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initializes the ElasticNetMethod instance.

        Args:
            dataloader (DataLoader): A data loader object which is expected to
                                     provide the necessary data for the method.

        This method initializes the base class and sets up the ElasticNetCV model.
        """
        super().__init__(dataloader)
        self.elastic = ElasticNetCV()

    def train(self) -> None:
        """
        Trains the Elastic Net Regression model with cross-validation.

        This method fits the ElasticNetCV model to the data provided by the
        dataloader. It assumes that the dataloader has attributes `X` for the
        input features and `y` for the target variable.
        """
        self.elastic.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained Elastic Net Regression model.

        Args:
            x (np.ndarray): The input data for which predictions are to be made.

        Returns:
            np.ndarray: The predicted values as a NumPy array. The output format
                        is dictated by scikit-learn's ElasticNetCV model.
        """
        return self.elastic.predict(x)
