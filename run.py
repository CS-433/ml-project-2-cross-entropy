"""Holds the code for reproducing the results of ridge regression model."""
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import config_parser
from data.dataset import Dataset
from data.dataloader import DataLoader
from data.feature.descriptor import DescriptorFeaturizer
from methods.ridge_method import RidgeMethod
from utils.visualize import visualize_energy

DIMER_FILE = 'dataset/xe2_50.xyz'
TRIMER_FILE = 'dataset/xe3_50.xyz'
RAND_TRIMER_FILE = 'dataset/xe3_dataset_dft.xyz'

N_DIMER_TRAIN = 40
N_TRIMER_TRAIN = 40
N_RAND_TRIMER_TRAIN = 5000

def fix_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.random.manual_seed(seed)

def rmse(y_true, y_pred, y_train):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.std(y_train)

def main():

    # Load data, calculate descriptors and split into train and validation
    featurizer = DescriptorFeaturizer()
    dimer_dataset = Dataset.from_file(DIMER_FILE, 2, featurizer)
    trimer_dataset = Dataset.from_file(TRIMER_FILE, 3, featurizer)
    rand_trimer_dataset = Dataset.from_file(RAND_TRIMER_FILE, 3, featurizer)

    dimer_train, dimer_val = dimer_dataset.split(
        [list(range(N_DIMER_TRAIN)),
        list(range(N_DIMER_TRAIN, len(dimer_dataset)))])
    trimer_train, trimer_val = trimer_dataset.split(
        [list(range(N_TRIMER_TRAIN)),
        list(range(N_TRIMER_TRAIN, len(dimer_dataset)))])
    rand_trimer_train, rand_trimer_val = rand_trimer_dataset.split(
        [list(range(N_RAND_TRIMER_TRAIN)),
        list(range(N_RAND_TRIMER_TRAIN, len(rand_trimer_dataset)))])
    train = DataLoader([dimer_train, trimer_train, rand_trimer_train])
    dimer_val = DataLoader([dimer_val])
    trimer_val = DataLoader([trimer_val])
    rand_trimer_val = DataLoader([rand_trimer_val])

    # Train
    method = RidgeMethod(train)
    method.train()

    # Predict and visualize
    dimer_energy = method.predict(dimer_dataset.X) * 2
    trimer_energy = method.predict(trimer_dataset.X) * 3
    fig, ax = visualize_energy(dimer_dataset, dimer_energy - dimer_energy[-1],
                               trimer_dataset, trimer_energy - trimer_energy[-1])
    ax[0].set_title('RidgeMethod')

    # Predict and calculate RMSE
    train_dimer_energy = method.predict(dimer_train.X) * 2
    train_trimer_energy = method.predict(trimer_train.X) * 3
    train_rand_trimer_energy = method.predict(rand_trimer_train.X) * 3
    val_dimer_energy = method.predict(dimer_val.X) * 2
    val_trimer_energy = method.predict(trimer_val.X) * 3
    val_rand_trimer_energy = method.predict(rand_trimer_val.X) * 3

    data = pd.DataFrame(columns=['train%RMSE$_{dimer}$', 'train%RMSE$_{trimer}$',
                                 'train%RMSE$_{rand_trimer}$', '%RMSE$_{dimer}$',
                                 '%RMSE$_{trimer}$', '%RMSE$_{rand_trimer}$'])

    data.loc['RidgeMethod'] = [
        rmse(train_dimer_energy, dimer_train.y * 2, train.y[:N_DIMER_TRAIN]),
        rmse(train_trimer_energy, trimer_train.y * 3, train.y[50:50+N_TRIMER_TRAIN]),
        rmse(train_rand_trimer_energy, rand_trimer_train.y * 3, train.y[100:100+N_RAND_TRIMER_TRAIN]),
        rmse(val_dimer_energy, dimer_val.y * 2, train.y[:N_DIMER_TRAIN]),
        rmse(val_trimer_energy, trimer_val.y * 3, train.y[50:50+N_TRIMER_TRAIN]),
        rmse(val_rand_trimer_energy, rand_trimer_val.y * 3, train.y[100:100+N_RAND_TRIMER_TRAIN])]

    print(data)
    plt.show()



if __name__ == '__main__':
    print("Loading config")
    parser = config_parser()
    args = parser.parse_args()
    fix_seed(args.seed)

    # start experiment
    main()