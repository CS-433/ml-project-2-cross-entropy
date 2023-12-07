import ase.io
import metatensor
import numpy as np
import scipy
from equisolve.utils import ase_to_tensormap

from rascaline import AtomicComposition, LodeSphericalExpansion
from rascaline.utils import PowerSpectrum
from radial_basis import KspaceRadialBasis


class Dataset:
    def __init__(self, path, num):
        super(Dataset).__init__()
        self.raw_data = ase.io.read(path, ":")

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


        descriptor_rs = rs_calculator.compute(self.raw_data)
        descriptor_rs = descriptor_rs.components_to_properties(["spherical_harmonics_m"])
        descriptor_rs = descriptor_rs.keys_to_properties(
            ["species_neighbor", "spherical_harmonics_l"]
        )
        descriptor_rs = descriptor_rs.keys_to_samples(["species_center"])
        descriptor_rs = metatensor.sum_over_samples(
            descriptor_rs, sample_names=["center", "species_center"]
        )

        descriptor_ps = ps_calculator.compute(self.raw_data)
        descriptor_ps = descriptor_ps.keys_to_samples(["species_center"])
        descriptor_ps = metatensor.sum_over_samples(
            descriptor_ps, sample_names=["center", "species_center"]
        )

        descriptor_co = co_calculator.compute(self.raw_data)
        descriptor_co = descriptor_co.keys_to_properties("species_center")

        if rs_ps:
            X = metatensor.join([descriptor_rs, descriptor_ps], axis="properties")
        else:
            X = metatensor.join([descriptor_rs], axis="properties")

        y = ase_to_tensormap(self.raw_data, energy="energy")
        self.X = X[0].values / num
        self.y = y[0].values / num

    def __iter__(self, to_tensor=False):
        return None


