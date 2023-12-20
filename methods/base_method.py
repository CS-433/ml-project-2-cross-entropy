import abc

import numpy as np

from data.dataloader import DataLoader


class BaseMethod:
    """
    An abstract base class representing a generic machine learning method.

    This class provides a structure for defining machine learning methods with
    mandatory implementation of `train` and `predict` methods, as well as
    an optional pre-processing step that is invoked during initialization.

    Attributes:
        dataloader (DataLoader): A data loader object which is used to load and
                          preprocess the data.

    Methods:
        train: An abstract method to be implemented for training the model.
        predict: An abstract method to be implemented for making predictions.
        pre_process: A method for pre-processing data, which can be overridden
                     by subclasses.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """
        Initializes the BaseMethod instance.

        Args:
            dataloader (DataLoader): A data loader object which is expected to
                              provide the necessary data for the method.

        The initialization includes a call to the `pre_process` method,
        allowing for data preprocessing steps to be performed.
        """
        self.pre_process(dataloader)
        self.dataloader = dataloader

    @abc.abstractmethod
    def train(self) -> None:
        """
        Abstract method for training the model.

        This method needs to be implemented in any subclass, defining
        the training procedure for the specific machine learning method.
        """
        pass

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method for making predictions.

        Args:
            x (np.ndarray): The input data for which predictions are to be made.

        Returns:
            np.ndarray: The output of the prediction, the format of which depends on
                 the specific implementation in the subclass.

        This method needs to be implemented in any subclass, defining
        the prediction procedure for the specific machine learning method.
        """
        pass

    def pre_process(self, dataloader: DataLoader) -> None:
        """
        Method for pre-processing data.

        Args:
            dataloader (DataLoader): The dataloader object containing the data to be
                              pre-processed.

        This method can be overridden by subclasses to include specific
        pre-processing steps required by the method.
        """
        pass
