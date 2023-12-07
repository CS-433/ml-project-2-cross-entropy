import numpy as np


class DataLoader:
    def __init__(self, datasets):
        super(DataLoader).__init__()

        self.X = np.vstack([dataset.X for dataset in datasets])
        self.y = np.vstack([dataset.y for dataset in datasets])
