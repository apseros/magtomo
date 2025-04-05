import magpack
from magtomo.Experiment import Experiment
import numpy as np
import logging
import functools
import multiprocessing
from scipy.linalg import block_diag
from scipy.ndimage import affine_transform


class Reconstruction(Experiment):
    """Performs a reconstruction from dichroic projections, inherits the properties that are specified by an
    :class:`magtomo.Experiment.Experiment`."""

    def __init__(self, magnetization, rotations, projections, pol=1j, iterations=10, mask=None, learning_parameter=10):
        """
        Parameters
        ----------
        magnetization : np.ndarray
            The initial guess for the magnetization.
        rotations : np.ndarray
            Stack of rotation matrices describing the sample orientations that were measured; shaped (n, 3, 3).
        projections : np.ndarray
            Stack of projections obtained at the corresponding `rot` value.
        pol : array_like (optional)
            Single polarization used for all projections, stack of polarizations matching the number of projections or
            stack of pairs of polarizations. Linear polarization represented using values from 0 to 180 (in degrees).
            Circular left and right polarization represented using complex ±1j.
        iterations : int (optional)
            Number of reconstruction iterations to perform.
        mask : np.ndarray (optional)
            Binary mask for the magnetization.
        learning_parameter : float (optional)
            The learning parameter for gradient descent.
        """
        super().__init__(magnetization, rotations, pol)
        self._error_metric = None
        self.iterations = iterations
        self.learning_parameter = learning_parameter
        self.projections = projections
        self.mask = mask
        self._error_norm = self.calc_error_norm()

    @property
    def projections(self):
        """The measured projections."""
        return self._projections

    @projections.setter
    def projections(self, value):
        if value.ndim != 3:
            raise ValueError("Projections must be 3-dimensional.")
        self._projections = value
        self.calc_error_norm()

    @property
    def learning_parameter(self):
        """The learning parameter for gradient descent."""
        return self.learning_parameter

    @learning_parameter.setter
    def learning_parameter(self, value):
        self._learning_parameter = value

    @property
    def iterations(self):
        """The number of reconstruction iterations."""
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        if value < 1:
            raise ValueError("Iterations must be greater than 0.")
        self._iterations = value
        if self._error_metric is None or self._error_metric.shape != self._iterations:
            self._error_metric = np.empty(self._iterations)

    @property
    def error_metric(self):
        """The error metric value at each iteration."""
        return self._error_metric

    @property
    def mask(self):
        """Binary mask for the magnetization."""
        return self._mask

    @mask.setter
    def mask(self, value: np.ndarray):
        if value is None:
            self._mask = np.ones_like(self._magnetization)
        elif value.shape == self.magnetization.shape[-3:]:
            self._mask = value
        else:
            raise ValueError("Mask must be same shape as magnetization.")

    def calc_error_norm(self):
        """Calculates a normalization factor (number of pixels in the projections) for scaling the error metric.

        Returns
        -------
        int
            The normalization factor (number of pixels in the projections).
        """
        return np.prod(self._projections.shape)

    def configuration_error(self, reference, axial=False):
        r"""Computes the error between the current reconstruction and a provided reference.

        Parameters
        ----------
        reference : np.ndarray
            Reference structure as a numpy array that will be compared with the current reconstruction.
        axial : bool (optional)
            Set to True if reconstructing orientation fields (e.g. linear dichroic projections).

        Returns
        -------
        float
            The configuration error.

        Notes
        -----
        The configuration error, :math:`\varepsilon_c` is evaluated for directional (vector) and axial (orientation)
        fields using:

        .. math::

            \varepsilon_c = \begin{cases} \vec{m}_r - \vec{m} & \text{directional} \\
            \min\left(\left\lvert \vec{m}_r - \vec{m} \right\rvert,
            \left\lvert \vec{m}_r + \vec{m} \right\rvert\right) & \text{axial}
            \end{cases}

        where :math:`\vec{m}_r` is the reconstruction and :math:`\vec{m}` is the reference.
        """
        if reference.shape != self._magnetization.shape:
            raise ValueError("Reference shape must match magnetization shape.")

        error = np.sum((self._magnetization - reference) ** 2, axis=0)
        if axial:
            error_branch = np.sum((self._magnetization + reference) ** 2, axis=0)
            error = np.where(error_branch < error, error_branch, error)
        return error

    def theta_error(self, reference: np.ndarray, axial: bool = False):
        r"""Computes the angular difference between the current reconstruction and a provided reference.

        Parameters
        ----------
        reference : np.ndarray
            Reference structure as a numpy array that will be compared with the current reconstruction.
        axial : bool (optional)
            Set to True if reconstructing orientation fields (e.g. linear dichroic projections).

        Returns
        -------
        float
            The angular error in radians.

        Notes
        -----
        The angular error, :math:`\varepsilon_\theta` is evaluated for directional (vector) and axial (orientation)
        fields using:

        .. math::

            \varepsilon_c =
            \begin{cases} \arccos\left( \frac{\vec{m_r} \cdot \vec{m}}{|\vec{m_r}||\vec{m}|} \right)
            & \text{directional} \\
            \arccos\left( \frac{\left\lvert \vec{m_r} \cdot \vec{m} \right\rvert}{|\vec{m_r}| |\vec{m}|} \right)
             & \text{axial}
            \end{cases}

        where :math:`\vec{m}_r` is the reconstruction and :math:`\vec{m}` is the reference.
        """
        return magpack.vectorop.angular_difference(self._magnetization, reference, axial=axial)

    def reconstruct(self):
        """Performs the reconstructions and updates the magnetization."""
        self._prepare()
        learning_param = self._learning_parameter

        for ii in range(self._iterations):
            logging.info(f"Currently on iteration {ii + 1}")

            # compute differences
            self.calculate_sinogram()
            difference = self._projections - self._sinogram
            current_error = 2 * np.sum(difference ** 2) / self._error_norm
            self._error_metric[ii] = current_error
            logging.info(f"Error metric: {current_error}")

            # evaluate gradients
            partial_gradient = functools.partial(gradient, self._magnetization, order=self._order)
            with multiprocessing.Pool() as p:
                grad = p.starmap(partial_gradient, zip(self._rotations, difference, self._pol))

            # apply gradient
            total_gradient = np.sum(grad, axis=0) / self._rotations.shape[0]
            dx = total_gradient / np.linalg.norm(total_gradient)
            self._magnetization += learning_param * dx * self._mask


    def _prepare(self):
        """Perform checks before initiating reconstruction"""
        if isinstance(self._pol, (int, float, complex)):
            self.pol = np.repeat(self._pol, self._rotations.shape[0])
        elif isinstance(self._pol, np.ndarray) and self._pol.shape[0] != self._rotations.shape[0]:
            raise ValueError("Number of polarisations must match number of rotations.")
        self.magnetization = self._magnetization * self._mask


def gradient(vf, rot, difference, pol, order):
    """Calculates the gradient for a specific orientation.

    Parameters
    ----------
    vf : np.ndarray
        The current guess for the vector or orientation field.
    rot : np.ndarray
        The rotation matrix describing the sample rotation
    difference : np.ndarray
        The difference between the projection from `vf` and the input.
    pol : np.ndarray
        The polarisation at which the projection was measured.
    order : int
        The interpolation order for performing rotations.

    Returns
    -------
    np.ndarray
        The gradient for this combination of inputs.
    """
    difference_field = difference[..., np.newaxis].repeat(vf.shape[-1], axis=-1) / vf.shape[-1]
    difference_field = rotate(difference_field, rot, order)

    if isinstance(pol, np.ndarray) and len(pol) == 2:
        return gradient(vf, rot, difference, pol[0], order) - gradient(vf, rot, difference, pol[1], order)
    elif isinstance(pol, np.ndarray):
        raise ValueError("Polarisations for dichroic input must be in pairs.")

    if isinstance(pol, complex) and pol.imag != 0:
        pol = pol.imag
        grad_x, grad_y, grad_z = rot[2] * pol
    else:
        pol = pol.real if isinstance(pol, complex) else pol
        electric_field = (np.cos(np.deg2rad(pol)), np.sin(np.deg2rad(pol)), 0)
        grad_x, grad_y, grad_z = np.einsum('ij,lk,jabc,i,l->kabc', rot, rot, vf, electric_field,
                                           electric_field) * 4
    grad_x = np.multiply(grad_x, difference_field)
    grad_y = np.multiply(grad_y, difference_field)
    grad_z = np.multiply(grad_z, difference_field)

    output_grad = np.stack([grad_x, grad_y, grad_z])
    return output_grad


def rotate(config, rot_matrix, order=1):
    """Applies an affine rotation described by the matrix rot_matrix about the center of the object.

    If the object has the shape (x, y, z) i.e. is three-dimensional, then the rotation is applied as expected.
    If the shape has the shape (n, x, y, z), i.e. multiple components need to be rotated, the rotation matrix is
    transformed into a 4x4 matrix given by [1, rot_matrix]

    Parameters
    ----------
    config : np.ndarray
        The object to be rotated. Must have shape (x, y, z) or (n, x, y, z)
    rot_matrix : np.ndarray
        The rotation matrix describing the operation to be applied.
    order : int
        The interpolation order of the affine transformation (0 - 5).

    Returns
    -------
    np.ndarray
        The rotated object (same shape as config).
    """
    config_shape = np.asarray(config.shape)

    if config.ndim == 4:
        rot_matrix = block_diag(1, rot_matrix)
    out_center = rot_matrix @ (config_shape - 1) / 2
    in_center = (config_shape - 1) / 2
    offset = in_center - out_center

    if 0 <= order <= 5:
        return affine_transform(config, rot_matrix, offset=offset, order=order)
    else:
        raise Exception("Interpolation order not recognised.")
