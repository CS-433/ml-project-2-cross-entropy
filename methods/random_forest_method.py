import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class RandomForestMethod(BaseMethod):
    """
    A concrete implementation of the BaseMethod for Random Forest Regression.

    This class extends the BaseMethod and implements the required `train` and `predict`
    methods using the RandomForestRegressor model from scikit-learn. RandomForestRegressor
    is an ensemble learning method, typically used for regression and classification tasks.

    Attributes:
        random_forest (RandomForestRegressor): An instance of the RandomForestRegressor model
                                               from scikit-learn.

    Methods:
        train: Implements the training procedure for the Random Forest Regression model.
        predict: Implements the prediction process using the trained Random Forest Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initializes the RandomForestMethod instance.

        Args:
            dataloader (DataLoader): A data loader object which is expected to
                                     provide the necessary data for the method.

        This method initializes the base class and sets up the RandomForestRegressor model.
        """
        super().__init__(dataloader)
        self.random_forest = RandomForestRegressor()

    def train(self) -> None:
        """
        Trains the Random Forest Regression model.

        This method fits the RandomForestRegressor model to the data provided by the
        dataloader. It assumes that the dataloader has attributes `X` for the
        input features and `y` for the target variable.
        """
        self.random_forest.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained Random Forest Regression model.

        Args:
            x (np.ndarray): The input data for which predictions are to be made.

        Returns:
            np.ndarray: The predicted values as a NumPy array. The output format
                        is dictated by scikit-learn's RandomForestRegressor model.
        """
        return self.random_forest.predict(x)
