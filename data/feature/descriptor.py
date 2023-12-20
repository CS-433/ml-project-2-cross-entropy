import os
import metatensor
import numpy as np
from equisolve.utils import ase_to_tensormap
from rascaline import AtomicComposition, LodeSphericalExpansion
from rascaline.utils import PowerSpectrum
from radial_basis import KspaceRadialBasis

from .feature_base import FeatureBase

class DescriptorFeaturizer(FeatureBase):

    def __init__(self,
                 max_radial: int = 6,
                 max_angular: int = 4,
                 cutoff: float = 3.,
                 atomic_gaussian_width: float = 1.,
                 rs_ps: bool = True):

        self.cutoff = cutoff
        self.max_radial = max_radial
        self.max_angular = max_angular
        self.atomic_gaussian_width = atomic_gaussian_width
        self.radial_basis = "monomial_spherical"
        self.rs_ps = rs_ps  # Use a rs and a ps or only a rs

        self._rs_calculator, self._co_calculator, self._ps_calculator = \
            self._config_calculator()

    def featurize(self, raw_data: AtomicComposition, file_name: str = None):

        X_tensor = file_name.replace(".xyz", "_x.npz")
        y_tensor = file_name.replace(".xyz", "_y.npz")

        if os.path.exists(X_tensor) and os.path.exists(y_tensor):
            X = metatensor.load(X_tensor)
            y = metatensor.load(y_tensor)

            return X[0].values, y[0].values
        
        descriptor_rs = self._rs_calculator.compute(raw_data)
        descriptor_rs = descriptor_rs.components_to_properties(["spherical_harmonics_m"])
        descriptor_rs = descriptor_rs.keys_to_properties(
            ["species_neighbor", "spherical_harmonics_l"]
        )
        descriptor_rs = descriptor_rs.keys_to_samples(["species_center"])
        descriptor_rs = metatensor.sum_over_samples(
            descriptor_rs, sample_names=["center", "species_center"]
        )

        descriptor_ps = self._ps_calculator.compute(raw_data)
        descriptor_ps = descriptor_ps.keys_to_samples(["species_center"])
        descriptor_ps = metatensor.sum_over_samples(
            descriptor_ps, sample_names=["center", "species_center"]
        )

        descriptor_co = self._co_calculator.compute(raw_data)
        descriptor_co = descriptor_co.keys_to_properties("species_center")

        if self.rs_ps:
            X = metatensor.join([descriptor_rs, descriptor_ps], axis="properties")
        else:
            X = metatensor.join([descriptor_rs], axis="properties")
        y = ase_to_tensormap(raw_data, energy="energy")

        metatensor.save(X_tensor, X)
        metatensor.save(y_tensor, y)

        return X[0].values, y[0].values

    def _config_calculator(self):
        lr_hypers_rs = {
            "cutoff": self.cutoff,
            "max_radial": self.max_radial,
            "max_angular": 0,
            "atomic_gaussian_width": self.atomic_gaussian_width,
            "center_atom_weight": 1.0,
            "potential_exponent": 6,
            "radial_basis": {self.radial_basis: {}},
        }

        lr_hypers_ps = {
            "cutoff": 3.0,
            "max_radial": self.max_radial,
            "max_angular": self.max_angular,
            "atomic_gaussian_width": self.atomic_gaussian_width,
            "center_atom_weight": 1.0,
            "potential_exponent": 4,
            "radial_basis": {self.radial_basis: {}},
        }

        orthonormalization_radius = self.cutoff
        k_cut = 1.2 * np.pi / self.atomic_gaussian_width

        rad_rs = KspaceRadialBasis(
            self.radial_basis,
            max_radial=self.max_radial,
            max_angular=0,
            projection_radius=self.cutoff,
            orthonormalization_radius=orthonormalization_radius,
        )

        lr_hypers_rs["radial_basis"] = rad_rs.spline_points(
            cutoff_radius=k_cut, requested_accuracy=1e-8
        )

        rad_ps = KspaceRadialBasis(
            self.radial_basis,
            max_radial=self.max_radial,
            max_angular=self.max_angular,
            projection_radius=self.cutoff,
            orthonormalization_radius=orthonormalization_radius,
        )

        lr_hypers_ps["radial_basis"] = rad_ps.spline_points(
            cutoff_radius=k_cut, requested_accuracy=1e-8
        )

        return LodeSphericalExpansion(**lr_hypers_rs), \
               AtomicComposition(per_structure=True), \
               PowerSpectrum(
                   LodeSphericalExpansion(**lr_hypers_ps),
                   LodeSphericalExpansion(**lr_hypers_ps),
               )
