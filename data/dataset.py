from dataclasses import dataclass
from typing import Union
import ase.io
import numpy as np
from data.feature.feature_base import FeatureBase

@dataclass
class Dataset:
    """A dataset class holding raw data and featurized data.
    Attributes:
        raw_data (list): The raw data
        X (np.ndarray): The featurized data
        y (np.ndarray): The energy data
        zero_point (float): The zero point energy
    """
    raw_data: list
    X: np.ndarray
    y: np.ndarray
    energy_base: float = 0

    def __post_init__(self):
        # sanity check
        assert isinstance(self.raw_data, list)
        assert len(self.raw_data) == len(self.X)
        assert len(self.raw_data) == len(self.y)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __iter__(self, to_tensor=False):
        return None

    @classmethod
    def from_file(cls, path, num, featurizer: FeatureBase, enegry_base=0):
        """Load data from a file.
        Args:
            path (str): The path to the file.
            num (int): The number of atoms in each frame of the dataset.
            featurizer (Featurizer): The featurizer.
        """
        raw_data = ase.io.read(path, ":")
        X, y = featurizer.featurize(raw_data, path)

        X = X / num
        y = y / num
        y -= enegry_base

        return cls(raw_data, X, y, enegry_base)

    def split(self, indexs: list):
        """Split the dataset.
        Args:
            indexs (list): The indexs to split the dataset.

        Returns:
            list: A list of datasets.
        
        Example:
            >>> train, val = dataset.split(
                    [list(range(10)),
                     list(range(10, 50))
                    ])
        """
        return [Dataset([self.raw_data[i] for i in index],
                        self.X[index],
                        self.y[index])
                        for index in indexs]


    def get_energy(self,
                   zero_point: Union[float, None] = None,
                   zero_point_idx: Union[int, None] = None
                   ) -> Union[np.ndarray, None]:
        """Get the energy from the raw data.
        Args:
            zero_point (Union[float, None], optional): The zero point energy. 
                Defaults to None. You can manully set it if you want to use it. 
                If you set zero_point_idx, zero_point will be ignored.
            zero_point_idx (Union[int, None], optional): The index of the data 
                you want to use as the zero point. Defaults to None."""
        energy = np.array([atoms.info["energy"] for atoms in self.raw_data])
        if zero_point is not None:
            zero_point_idx = None
            energy -= zero_point
        if zero_point_idx is not None:
            energy -= energy[zero_point_idx]

        return energy

    def get_distance(self) -> Union[np.ndarray, None]:
        """Get the distance from the raw data."""

        try:
            distances = np.array(
                [atoms.info["distance"] for atoms in self.raw_data]
            )
        except KeyError:
            print('No distance')
            return None

        return distances
