import ase.io
import metatensor
import numpy as np
import scipy
from scipy.spatial import distance_matrix
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import metatensor
from rascaline import AtomicComposition, LodeSphericalExpansion
from rascaline.utils import PowerSpectrum
from radial_basis import KspaceRadialBasis
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from equisolve.utils.convert import ase_to_tensormap


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


def calc_dimer_trimer_energies(frames, energy_inter):
    dimer_trimer_energies = np.zeros(len(frames))

    for i, atoms in enumerate(frames):
        dist_mat = distance_matrix(atoms.positions, atoms.positions)
        # print(dist_mat)
        pair_distances = dist_mat[np.triu_indices(len(atoms), k=1)]
        # if np.isnan(pair_distances):
        pred_dimer = energy_inter(pair_distances)
        # print(pred_dimer)
        pred_dimer[np.isnan(pred_dimer)] = energy_inter(14)
        dimer_trimer_energies[i] = np.sum(pred_dimer)
        # if np.isnan(dimer_trimer_energies[i]):
        # print(pred_dimer)

    return dimer_trimer_energies


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


if __name__ == '__main__':
    ori_frames_datasets = ase.io.read("xe3_dataset_dft.xyz", ":")
    frames_datasets = ori_frames_datasets  # []
    # for atoms in ori_frames_datasets:
    #    dist_mat = distance_matrix(atoms.positions, atoms.positions)
    #    if np.all(dist_mat <= 14):
    #        frames_datasets.append(atoms)
    dimer_datasets = frames_datasets[:50]
    trimer_datasets = frames_datasets[50:100]
    random_trimer_datasets = frames_datasets[100:]
    n_total_random = len(random_trimer_datasets)

    d = []
    for i, atoms in enumerate(random_trimer_datasets):
        dist_mat = distance_matrix(atoms.positions, atoms.positions)
        # print(dist_mat)
        pair_distances = dist_mat[np.triu_indices(len(atoms), k=1)]
        d.append(pair_distances)
    d = np.concatenate(d)

    n_dimer = 40
    n_trimer = 40
    dimer_idx = np.random.choice(list(range(50)), n_dimer, replace=False)
    trimer_idx = np.random.choice(list(range(50, 100)), n_trimer, replace=False)
    # dimer_idx = np.arange(n_dimer)
    # trimer_idx = np.arange(50, n_trimer + 50)

    # Find where Xenon3 starts
    # for i, atoms in enumerate(frames_dataset):
    #    if len(atoms) == 3:
    #        split = i
    #        break

    # Holds codes dealing with dimer and equilateral trimer energy

    distances_dimers = np.array(
        [atoms.info["distance"] for atoms in dimer_datasets]
    )
    energies_dimers = np.array([atoms.info["energy"] for atoms in dimer_datasets])

    distances_trimers = np.array(
        [atoms.info["distance"] for atoms in trimer_datasets]
    )
    energies_trimers = np.array([atoms.info["energy"] for atoms in trimer_datasets])

    energies_random_trimers = np.array([atoms.info["energy"] for atoms in random_trimer_datasets])

    dimer_energy_inter = scipy.interpolate.CubicSpline(
        distances_dimers, energies_dimers.copy(), extrapolate=False
    )

    energies_trimers_drimers = calc_dimer_trimer_energies(
        trimer_datasets, dimer_energy_inter
    )

    trimer_zero_point = energies_trimers[-1]
    diff_zero_point = energies_trimers_drimers[-1]
    energies_dimers -= energies_dimers[-1]
    energies_trimers -= trimer_zero_point
    energies_trimers_drimers -= diff_zero_point

    ratio = 0.01
    n_random_trimer = int(n_total_random * ratio)
    random_trimer_idx = np.random.choice(list(range(100, 100 + len(random_trimer_datasets))), n_random_trimer,
                                         replace=False)
    mask = np.full(len(frames_datasets), False)
    mask[dimer_idx] = True
    mask[trimer_idx] = True
    mask[random_trimer_idx] = True

    train_frames_dataset, test_frames_dataset = get_train_test(mask)

    energies_random_trimers_dimers = calc_dimer_trimer_energies(
        random_trimer_datasets, dimer_energy_inter
    )

    energies_random_trimers -= trimer_zero_point
    energies_random_trimers_dimers -= diff_zero_point
    print("")

    # Train model

    cutoff = 3.0
    max_radial = 6
    max_angular = 4
    atomic_gaussian_width = 1.0
    radial_basis = "monomial_spherical"

    lr_hypers_rs = {
        "cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": 0,
        "atomic_gaussian_width": atomic_gaussian_width,
        "center_atom_weight": 1.0,
        "potential_exponent": 6,
        "radial_basis": {radial_basis: {}},
    }

    lr_hypers_ps = {
        "cutoff": 3.0,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "atomic_gaussian_width": atomic_gaussian_width,
        "center_atom_weight": 1.0,
        "potential_exponent": 4,
        "radial_basis": {radial_basis: {}},
    }

    orthonormalization_radius = cutoff
    k_cut = 1.2 * np.pi / atomic_gaussian_width

    rad_rs = KspaceRadialBasis(
        radial_basis,
        max_radial=max_radial,
        max_angular=0,
        projection_radius=cutoff,
        orthonormalization_radius=orthonormalization_radius,
    )

    lr_hypers_rs["radial_basis"] = rad_rs.spline_points(
        cutoff_radius=k_cut, requested_accuracy=1e-8
    )

    rad_ps = KspaceRadialBasis(
        radial_basis,
        max_radial=max_radial,
        max_angular=max_angular,
        projection_radius=cutoff,
        orthonormalization_radius=orthonormalization_radius,
    )

    lr_hypers_ps["radial_basis"] = rad_ps.spline_points(
        cutoff_radius=k_cut, requested_accuracy=1e-8
    )

    rs_calculator = LodeSphericalExpansion(**lr_hypers_rs)

    co_calculator = AtomicComposition(per_structure=True)

    ps_calculator = PowerSpectrum(
        LodeSphericalExpansion(**lr_hypers_ps),
        LodeSphericalExpansion(**lr_hypers_ps),
    )

    training_cutoff = 7.5
    rs_ps = True  # Use a rs and a ps or only a rs

    frames_fit = frames_datasets

    descriptor_rs = rs_calculator.compute(frames_fit)
    descriptor_rs = descriptor_rs.components_to_properties(["spherical_harmonics_m"])
    descriptor_rs = descriptor_rs.keys_to_properties(
        ["species_neighbor", "spherical_harmonics_l"]
    )
    descriptor_rs = descriptor_rs.keys_to_samples(["species_center"])
    descriptor_rs = metatensor.sum_over_samples(
        descriptor_rs, sample_names=["center", "species_center"]
    )

    descriptor_ps = ps_calculator.compute(frames_fit)
    descriptor_ps = descriptor_ps.keys_to_samples(["species_center"])
    descriptor_ps = metatensor.sum_over_samples(
        descriptor_ps, sample_names=["center", "species_center"]
    )

    descriptor_co = co_calculator.compute(frames_fit)
    descriptor_co = descriptor_co.keys_to_properties("species_center")

    if rs_ps:
        X = metatensor.join([descriptor_rs, descriptor_ps], axis="properties")
    else:
        X = metatensor.join([descriptor_rs], axis="properties")

    y = ase_to_tensormap(frames_fit, energy="energy")
    X[0].values[y[0].values.flatten() > -600000, :] /= 2
    X[0].values[y[0].values.flatten() < -600000, :] /= 3

    # ratio = 0.8
    # train_split = int(ratio * X[0].values.shape[0])
    # train_split = 700
    # idx_train = np.arange(train_split)[np.newaxis, :].T
    # idx_test = np.arange(train_split, X[0].values.shape[0])[np.newaxis, :].T

    # for i, atoms in enumerate(frames_fit):
    #    delta_distance = atoms.info["distance"]
    #    if delta_distance <= training_cutoff:
    #        idx_train += [[i]]
    #    else:
    #        idx_test += [[i]]
    #
    # idx_train = np.array(idx_train)
    # idx_test = np.array(idx_test)
    idx_train = np.array(train_frames_dataset)[np.newaxis, :].T
    idx_test = np.array(test_frames_dataset)[np.newaxis, :].T
    X_train, X_test, y_train, y_test = split(X, y, idx_train, idx_test)

    # COMPOSITION BASELINING
    # Only baseline if we have dimers and trimers
    rcv = RidgeCV(alphas=np.geomspace(1e-18, 1e2, 14), fit_intercept=False)
    rcv.fit(descriptor_co[0].values, y[0].values)
    print(rcv.alpha_)
    yp_base = rcv.predict(descriptor_co[0].values).flatten()

    y_diff = y[0].values.flatten() - yp_base
    y_diff_train = y_diff[idx_train].flatten()

    y_train[0].values[y_train[0].values > -600000] /= 2
    y_train[0].values[y_train[0].values < -600000] /= 3
    # y_train[y_train > -600000] /= 2
    # y_train[y_train < -600000] /= 3

    # if we baseline the energies we should not have to fit the intercept
    clf = RidgeCV(alphas=np.geomspace(1e-12, 1e2, 32), fit_intercept=True)
    clf.fit(X_train[0].values, y_train[0].values.flatten())
    # clf.fit(X_train, y_train)
    # clf.fit(X_train[0].values, y_diff_train)
    print(clf.alpha_)

    y_diff_pred = clf.predict(X[0].values).flatten()
    y_pred = np.copy(y_diff_pred)
    y_pred[:50] *= 2
    y_pred[50:] *= 3
    # y_pred = y_diff_pred.copy() + yp_base

    y_pred_dimers = y_pred[:50]
    y_pred_trimers = y_pred[50:]

    y_pred_inter = scipy.interpolate.CubicSpline(
        distances_dimers, y_pred_dimers, extrapolate=False
    )
    y_pred_dimers -= y_pred_dimers[-1]
    y_pred_trimers -= y_pred_trimers[49]
    y_dimer_trimer = calc_dimer_trimer_energies(frames_datasets[50:], y_pred_inter)
    y_dimer_trimer -= y_dimer_trimer[49]

    rmse_dimer = mean_squared_error(
        y_pred_dimers[np.array(test_frames_dataset)[:50 - n_dimer]],
        energies_dimers[np.array(test_frames_dataset)[:50 - n_dimer]], squared=False
    ) / np.std(y_pred_dimers[train_frames_dataset[:n_dimer]])

    rmse_trimer = mean_squared_error(
        y_pred_trimers[np.array(test_frames_dataset)[50 - n_trimer:] - 50],
        np.concatenate([energies_trimers, energies_random_trimers])[np.array(test_frames_dataset)[50 - n_trimer:] - 50],
        squared=False
    ) / np.std(np.concatenate([y_pred_dimers, y_pred_trimers])[train_frames_dataset])

    rmse_diff = mean_squared_error(
        np.concatenate([energies_trimers, energies_random_trimers]) - np.concatenate(
            [energies_trimers_drimers, energies_random_trimers_dimers]),
        y_pred_trimers - y_dimer_trimer, squared=False
    ) / np.std((y_pred_trimers - y_dimer_trimer)[np.array(train_frames_dataset)[50 - n_dimer:] - 50])

    outname_suffix = f"dimers{'+trimers' * True}_rs{'+ps' * rs_ps}"

    if rs_ps:
        subplotlabels = "cd"
    else:
        subplotlabels = "ab"
