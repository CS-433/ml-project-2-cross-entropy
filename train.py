import random

import numpy as np
import torch

from config import config_parser
from data.dataloader import DataLoader
from data.dataset import Dataset
from data.feature import Featurizer
from methods.mlp_method import MlpMethod
from visualize import visualize_energy


def main(args):
    featurizer = Featurizer()
    dimer_dataset = Dataset.from_file('xe2_50.xyz', 2, featurizer)
    trimer_dataset = Dataset.from_file('xe3_50.xyz', 3, featurizer)
    rand_trimer_dataset = Dataset.from_file('xe3_dataset_dft.xyz', 3, featurizer)

    dimer_train, dimer_val = dimer_dataset.split(
        [list(range(40)),
         list(range(40, 50))])
    trimer_train, trimer_val = trimer_dataset.split(
        [list(range(40)),
         list(range(40, 50))])
    rand_trimer_train, rand_trimer_val = rand_trimer_dataset.split(
        [[],
         list(range(len(rand_trimer_dataset)))])

    train = DataLoader([dimer_train, trimer_train, rand_trimer_train])
    val = DataLoader([dimer_val, trimer_val, rand_trimer_val])

    method = MlpMethod(train, epochs=300)

    method.train()

    dimer_energy = method.predict(dimer_dataset.X) * 2
    trimer_energy = method.predict(trimer_dataset.X) * 3

    visualize_energy(dimer_dataset, dimer_energy - dimer_energy[-1], trimer_dataset, trimer_energy - trimer_energy[-1])


if __name__ == '__main__':
    # load config
    print("Loading config")
    parser = config_parser()
    args = parser.parse_args()

    # Fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)

    # start experiment
    main(args)
