from typing import Union
import ase.io
import numpy as np
from equisolve.utils import ase_to_tensormap
from data.feature import Featurizer


class Dataset:
    def __init__(self, path, num, featurizer: Featurizer):
        super(Dataset).__init__()
        self.raw_data = ase.io.read(path, ":")

        X = featurizer.featurize(self.raw_data)
        y = ase_to_tensormap(self.raw_data, energy="energy")
        self.X = X[0].values / num
        self.y = y[0].values / num

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __iter__(self, to_tensor=False):
        return None

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
