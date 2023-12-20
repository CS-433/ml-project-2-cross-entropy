import numpy as np
from equisolve.utils import ase_to_tensormap
from .feature_base import FeatureBase

class CoordinateFeaturizer(FeatureBase):
    def featurize(self, raw_data, file: str = None):
        X = np.array([data.positions for data in raw_data])
        y = ase_to_tensormap(raw_data, energy="energy")

        return X, y[0].values
