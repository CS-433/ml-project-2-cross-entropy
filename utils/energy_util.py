import ase
import numpy as np
from scipy.spatial import distance_matrix

L = 40  # Å
CELL = L * np.eye(3)

def dist_pbc(pos1, pos2):
    """
    Calculate the distance between two points in a periodic boundary condition system.

    Parameters:
        pos1 (numpy.ndarray): The coordinates of the first point.
        pos2 (numpy.ndarray): The coordinates of the second point.

    Returns:
        float: The distance between the two points.
    """
    return ase.geometry.get_distances(pos1, pos2, cell=CELL, pbc=True)[1]

def calc_pair_distances(atoms: ase.Atoms):
    """
    Calculate the distances between pairs of atoms in an ASE Atoms object.

    Parameters:
        atoms (ase.Atoms): The ASE Atoms object containing the atoms.

    Returns:
        numpy.ndarray: An array containing the distances between pairs of atoms.
    """

    dist = []
    for i, atom_i in enumerate(atoms[:-1]):
        for atom_j in atoms[i+1:]:
            dist.append(dist_pbc(atom_i.position, atom_j.position)[0])

    return np.array(dist)

def calc_dimer_trimer_energies(frames, energy_inter):
    dimer_trimer_energies = np.zeros(len(frames))

    for i, atoms in enumerate(frames):
        pair_distances = calc_pair_distances(atoms)
        # print(dist_mat)
        # if np.isnan(pair_distances):
        pred_dimer = energy_inter(pair_distances)
        # print(pred_dimer)
        pred_dimer[np.isnan(pred_dimer)] = energy_inter(14)
        dimer_trimer_energies[i] = np.sum(pred_dimer)
        # if np.isnan(dimer_trimer_energies[i]):
        # print(pred_dimer)

    return dimer_trimer_energies