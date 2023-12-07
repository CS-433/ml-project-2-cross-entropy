import ase.io
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

    def __iter__(self, to_tensor=False):
        return None


