import random
import ase.io
import metatensor
import numpy as np
import scipy
import torch
from scipy.spatial import distance_matrix
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import metatensor
from rascaline import AtomicComposition, LodeSphericalExpansion
from rascaline.utils import PowerSpectrum

from config import config_parser
from data.dataloader import DataLoader
from data.dataset import Dataset
from data.feature import Featurizer
from methods.mlp_method import MlpMethod
from radial_basis import KspaceRadialBasis
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from equisolve.utils.convert import ase_to_tensormap

from utils.energy_util import calc_dimer_trimer_energies
from visualize import visualize_energy


def get_train_test(mask):
    train_frames_dataset = []
    test_frames_dataset = []
    for idx in range(len(mask)):
        if mask[idx]:
            train_frames_dataset.append(idx)
        else:
            test_frames_dataset.append(idx)
    train_frames_dataset = np.array(train_frames_dataset)
    test_frames_dataset = np.array(test_frames_dataset)

    return train_frames_dataset, test_frames_dataset



def split(X, y, idx_train, idx_test):
    samples_train = metatensor.Labels(["structure"], idx_train)
    samples_test = metatensor.Labels(["structure"], idx_test)

    split_args = {
        "axis": "samples",
        "grouped_labels": [
            samples_train,
            samples_test,
        ],
    }

    X_train, X_test = metatensor.split(X, **split_args)
    y_train, y_test = metatensor.split(y, **split_args)

    return X_train, X_test, y_train, y_test



def main(args):
    # frames_datasets = ase.io.read("xe3_dataset_dft.xyz", ":")
    # frames_datasets = ori_frames_datasets  # []
    # dimer_datasets = frames_datasets[:50]
    # trimer_datasets = frames_datasets[50:100]
    # random_trimer_datasets = frames_datasets[100:]
    # n_total_random = len(random_trimer_datasets)
    #
    # n_dimer = 40
    # n_trimer = 40
    # dimer_idx = np.random.choice(list(range(50)), n_dimer, replace=False)
    # trimer_idx = np.random.choice(list(range(50, 100)), n_trimer, replace=False)
    # # dimer_idx = np.arange(n_dimer)
    # # trimer_idx = np.arange(50, n_trimer + 50)
    #
    # # Find where Xenon3 starts
    # # for i, atoms in enumerate(frames_dataset):
    # #    if len(atoms) == 3:
    # #        split = i
    # #        break
    #
    # # Holds codes dealing with dimer and equilateral trimer energy
    # distances_dimers = np.array(
    #     [atoms.info["distance"] for atoms in dimer_datasets]
    # )
    # energies_dimers = np.array([atoms.info["energy"] for atoms in dimer_datasets])
    #
    # distances_trimers = np.array(
    #     [atoms.info["distance"] for atoms in trimer_datasets]
    # )
    # energies_trimers = np.array([atoms.info["energy"] for atoms in trimer_datasets])
    #
    # energies_random_trimers = np.array([atoms.info["energy"] for atoms in random_trimer_datasets])
    #
    # dimer_energy_inter = scipy.interpolate.CubicSpline(
    #     distances_dimers, energies_dimers.copy(), extrapolate=False
    # )
    #
    # energies_trimers_drimers = calc_dimer_trimer_energies(
    #     trimer_datasets, dimer_energy_inter
    # )
    #
    # trimer_zero_point = energies_trimers[-1]
    # diff_zero_point = energies_trimers_drimers[-1]
    # energies_dimers -= energies_dimers[-1]
    # energies_trimers -= trimer_zero_point
    # energies_trimers_drimers -= diff_zero_point
    #
    # ratio = 0.01
    # n_random_trimer = int(n_total_random * ratio)
    # random_trimer_idx = np.random.choice(list(range(100, 100 + len(random_trimer_datasets))), n_random_trimer,
    #                                      replace=False)
    # mask = np.full(len(frames_datasets), False)
    # mask[dimer_idx] = True
    # mask[trimer_idx] = True
    # mask[random_trimer_idx] = True
    #
    # train_frames_dataset, test_frames_dataset = get_train_test(mask)
    #
    # energies_random_trimers_dimers = calc_dimer_trimer_energies(
    #     random_trimer_datasets, dimer_energy_inter
    # )
    #
    # energies_random_trimers -= trimer_zero_point
    # energies_random_trimers_dimers -= diff_zero_point
    # print("")
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

    # idx_train = np.array(train_frames_dataset)[np.newaxis, :].T
    # idx_test = np.array(test_frames_dataset)[np.newaxis, :].T
    # X_train, X_test, y_train, y_test = split(X, y, idx_train, idx_test)

    # COMPOSITION BASELINING
    # Only baseline if we have dimers and trimers
    # rcv = RidgeCV(alphas=np.geomspace(1e-18, 1e2, 14), fit_intercept=False)
    # rcv.fit(descriptor_co[0].values, y[0].values)
    # print(rcv.alpha_)
    # yp_base = rcv.predict(descriptor_co[0].values).flatten()
    #
    # y_diff = y[0].values.flatten() - yp_base
    # y_diff_train = y_diff[idx_train].flatten()

    # y_train[0].values[y_train[0].values > -600000] /= 2
    # y_train[0].values[y_train[0].values < -600000] /= 3
    # y_train[y_train > -600000] /= 2
    # y_train[y_train < -600000] /= 3

    # if we baseline the energies we should not have to fit the intercept
    # clf = RidgeCV(alphas=np.geomspace(1e-12, 1e2, 32), fit_intercept=True)
    # clf.fit(dataloader.X, dataloader.y)
    # clf.fit(X_train, y_train)
    # clf.fit(X_train[0].values, y_diff_train)
    # print(clf.alpha_)
    #
    # y_diff_pred = clf.predict(dataloader.X)
    # y_pred = np.copy(y_diff_pred)
    # y_pred[:50] *= 2
    # y_pred[50:] *= 3
    # y_pred = y_diff_pred.copy() + yp_base

    # y_pred_dimers = y_pred[:50]
    # y_pred_trimers = y_pred[50:]
    #
    # y_pred_inter = scipy.interpolate.CubicSpline(
    #     distances_dimers, y_pred_dimers, extrapolate=False
    # )
    # y_pred_dimers -= y_pred_dimers[-1]
    # y_pred_trimers -= y_pred_trimers[49]
    # y_dimer_trimer = calc_dimer_trimer_energies(frames_datasets[50:], y_pred_inter)
    # y_dimer_trimer -= y_dimer_trimer[49]
    #
    # rmse_dimer = mean_squared_error(
    #     y_pred_dimers[np.array(test_frames_dataset)[:50 - n_dimer]],
    #     energies_dimers[np.array(test_frames_dataset)[:50 - n_dimer]], squared=False
    # ) / np.std(y_pred_dimers[train_frames_dataset[:n_dimer]])
    #
    # rmse_trimer = mean_squared_error(
    #     y_pred_trimers[np.array(test_frames_dataset)[50 - n_trimer:] - 50],
    #     np.concatenate([energies_trimers, energies_random_trimers])[np.array(test_frames_dataset)[50 - n_trimer:] - 50],
    #     squared=False
    # ) / np.std(np.concatenate([y_pred_dimers, y_pred_trimers])[train_frames_dataset])
    #
    # rmse_diff = mean_squared_error(
    #     np.concatenate([energies_trimers, energies_random_trimers]) - np.concatenate(
    #         [energies_trimers_drimers, energies_random_trimers_dimers]),
    #     y_pred_trimers - y_dimer_trimer, squared=False
    # ) / np.std((y_pred_trimers - y_dimer_trimer)[np.array(train_frames_dataset)[50 - n_dimer:] - 50])
    #
    # outname_suffix = f"dimers{'+trimers' * True}_rs{'+ps' * rs_ps}"
    #
    # if rs_ps:
    #     subplotlabels = "cd"
    # else:
    #     subplotlabels = "ab"


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
