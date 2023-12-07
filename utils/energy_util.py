import numpy as np
from scipy.spatial import distance_matrix


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