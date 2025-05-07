import functools
import logging
import numpy as np
import multiprocessing
from typing import Union, List
from ThreeDViewer.image import plot_3d
from magpack.rotations import rotate_scalar_field

Number = Union[int, float, complex]


class Experiment:
    """Defines an experimental setup for performing vector or orientation tomography."""

    def __init__(self, magnetization, rotations, pol=1j, order=1):
        """
        Parameters
        ----------
        magnetization : np.ndarray
            The magnetic configuration to calculate projections from.
        rotations : np.ndarray
            Stack of rotation matrices describing the sample orientations that were measured; shaped (n, 3, 3).
        pol : array_like (optional)
            Single polarization used for all projections, stack of polarizations matching the number of projections or
            stack of pairs of polarizations. Linear polarization represented using values from 0 to 180 (in degrees).
            Circular left and right polarization represented using complex ±1j.
        order : int
            The interpolation order (0 - 5) for performing rotations.
        """
        self.magnetization = magnetization
        self.rotations = rotations
        self.pol = pol
        self.order = order
        self._sinogram = None

    @property
    def rotations(self):
        """The stack of measured orientation described by rotation matrices."""
        return self._rotations

    @rotations.setter
    def rotations(self, value):
        if value.ndim == 2:
            value = value[np.newaxis, ...]
        if value.shape[-1] != value.shape[-2]:
            raise ValueError("Rotation matrices must be square.")
        if self._magnetization is not None and self._magnetization.shape[0] != value.shape[-1]:
            raise ValueError("Rotation matrix does not match number of field components.")
        self._rotations = value

    @property
    def magnetization(self):
        """The magnetic configuration of the sample."""
        return np.array(self._magnetization)

    @magnetization.setter
    def magnetization(self, value: np.ndarray):
        if value.ndim > 5:
            logging.warning('Magnetization array should be 4-dimensional with shape (3, nx, ny, nz) or 5 dimensional'
                            'with shape (3, 3, nx, ny, nz.')
        if value.shape[0] != 3:
            logging.warning('Magnetization should have three components.')

        self._magnetization = value

    @property
    def pol(self):
        """The polarization of the incident beam."""
        return self._pol

    @pol.setter
    def pol(self, value: Union[Number, List[Number], np.ndarray]):
        if isinstance(value, list):
            value = np.asarray(value)
        self._pol = value

    @property
    def order(self):
        """The interpolation order for performing rotations."""
        return self._order

    @order.setter
    def order(self, value: int):
        if value not in range(5):
            self._order = value
        else:
            self._order = 1

    @property
    def sinogram(self):
        """The calculated sinogram."""
        return self._sinogram

    def calculate_sinogram(self):
        """Calculates the sinogram."""
        if isinstance(self._pol, (int, float, complex)):
            self._pol = np.repeat(self._pol, self._rotations.shape[0])

        if np.all(self._pol.imag == 0):
            proj_method = tensor_projection
            if self.magnetization.shape[:2] != (3, 3):
                tensor_mag = np.einsum('iabc,jabc->ijabc', self._magnetization, self._magnetization)
                self.magnetization = tensor_mag
        else:
            proj_method = projection

        partial_projection = functools.partial(proj_method, self._magnetization, order=self._order)
        with multiprocessing.Pool() as p:
            proj = p.starmap(partial_projection, zip(self._rotations, self._pol))
        self._sinogram = np.asarray(proj)

    def plot_sinogram(self, cmap='Spectral'):
        """Plots the calculated sinogram."""
        plot_3d(self._sinogram, init_take=0, cmap=cmap, axes_names=(r"$\theta$", "x", "y"), title="Sinogram")


def tensor_projection(tensor_field, rot_matrix, pol=0, order=1):
    """Calculation of a single projection using tensor notation

    :param tensor_field:
    :param rot_matrix:
    :param pol:
    :param order:
    :return:
    """
    field_copy = np.copy(tensor_field)

    electric_field = (np.cos(np.deg2rad(pol)), np.sin(np.deg2rad(pol)), 0)
    proj = np.einsum('i,ij,jlabc,kl,k->abc',electric_field, rot_matrix, field_copy, rot_matrix,
                     electric_field)

    proj = rotate_scalar_field(proj, rot_matrix, order=order)
    return np.sum(proj, axis=2)


def projection(field, rot_matrix, pol=0, order = 1):
    """Calculation of a single projection.

    Parameters
    ----------
    field : np.ndarray
        The magnetization field.
    rot_matrix : np.ndarray
        A single rotation matrix.
    pol : np.ndarray (optional)
        Single polarization or a pair of polarizations (if the difference of projections is desired). Linear
        polarization represented using values from 0 to 180 (in degrees) and circular polarizations represented with
        +1j and -1j (complex unit values)
    order : int (optional)
        The interpolation order (0 - 5) for performing rotations.

    Returns
    -------
    np.ndarray
        The projection of the field at the orientation described by `rot_matrix` and polarization `pol`.
    """
    field_copy = np.copy(field)

    if isinstance(pol, np.ndarray) and len(pol) == 2:
        return projection(field, rot_matrix, pol[0], order) - projection(field, rot_matrix, pol[1], order)
    elif isinstance(pol, np.ndarray):
        raise ValueError("Polarisations for dichroic input must be in pairs.")

    if isinstance(pol, complex) and pol.imag != 0:
        pol = pol.imag
        proj = np.einsum('j,jabc -> abc', rot_matrix[2], field_copy) * pol
    else:
        pol = pol.real if isinstance(pol, complex) else pol
        electric_field = (np.cos(np.deg2rad(pol)), np.sin(np.deg2rad(pol)), 0)
        proj = np.einsum('ij,jabc,i->abc', rot_matrix, field_copy, electric_field) ** 2

    proj = rotate_scalar_field(proj, rot_matrix, order=order)
    return np.sum(proj, axis=2)
