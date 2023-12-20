import numpy as np
from sklearn.tree import DecisionTreeRegressor

from data.dataloader import DataLoader
from methods.base_method import BaseMethod


class DecisionTreeMethod(BaseMethod):
    """
        A concrete implementation of the BaseMethod for Decision Tree Regression.

        This class extends the BaseMethod and implements the required `train` and
        `predict` methods using the DecisionTreeRegressor model from scikit-learn.

        Attributes:
            decision_tree (DecisionTreeRegressor): An instance of the DecisionTreeRegressor
                                                   model from scikit-learn.

        Methods:
            train: Implements the training procedure for the Decision Tree Regression model.
            predict: Implements the prediction process using the trained Decision Tree Regression model.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
            Initializes the DecisionTreeMethod instance.

            Args:
                dataloader (DataLoader): A data loader object which is expected to
                                provide the necessary data for the method.

            This method initializes the base class and sets up the DecisionTreeRegressor model.
        """
        super().__init__(dataloader)
        self.decision_tree = DecisionTreeRegressor()

    def train(self) -> None:
        """
            Trains the Decision Tree Regression model.

            This method fits the DecisionTreeRegressor model to the data provided by the
            dataloader. It assumes that the dataloader has attributes `X` for the
            input features and `y` for the target variable.
        """
        self.decision_tree.fit(self.dataloader.X, self.dataloader.y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
            Makes predictions using the trained Decision Tree Regression model.

            Args:
                x (np.ndarray): The input data for which predictions are to be made.

            Returns:
                np.ndarray: The predicted values. The format and type of the output
                        depend on the implementation in scikit-learn's DecisionTreeRegressor model.
        """
        return self.decision_tree.predict(x)
