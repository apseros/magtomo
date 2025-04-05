import logging
import numpy as np
import magpack.rotations as rtn
import ThreeDViewer.data
from magtomo.Experiment import Experiment
from magtomo.Reconstruction import Reconstruction
from ThreeDViewer.image import plot_3d
from importlib import resources
from magtomo.radon import radon, inv_radon
from magpack import vectorop, structures, io

def vector_tomo(angles, tilts, kind='CD', it=10, learning=30):
    """Examples of vector and orientation tomography.

    Parameters
    ----------
    angles : np.ndarray | list of float
        The tomographic rotation angles.
    tilts : np.ndarray | list of float
        The tomographic tilt angles.
    kind : {'CD', 'LD'}
        Vector (CD) or Orientation (LD) tomography options.
    it : int
        The number of gradient descent iterations to evaluate.
    learning : float
        The learning parameter for gradient descent.
    """

    # specify a structure
    struct = _get_structure()
    mask = vectorop.magnitude(struct) > 0

    # get rotation matrices
    rot = rtn.tomo_rot(angles, tilts)
    # get polarizations for each rotation
    pol_a = np.repeat(-45 if kind == 'LD' else 1j, rot.shape[0])
    pol_b = np.repeat(+45 if kind == 'LD' else -1j, rot.shape[0])

    # double rotations to match the polarizations
    pol = np.hstack((pol_a, pol_b))
    rot = np.concatenate((rot, rot), axis=0)
    # initialize the experiment class with the structure, rotation matrices and polarizations
    exp = Experiment(magnetization=struct, rotations=rot, pol=pol)
    # calculate and sinogram that will be used as input
    exp.calculate_sinogram()
    # plot the sinogram
    cmap = 'jet' if kind == 'LD' else 'Spectral'
    exp.plot_sinogram(cmap=cmap)
    # these projections will be used to attempt a reconstruction
    projections = exp.sinogram
    # mask the sample for faster convergence (can also be derived from a scalar reconstruction)

    # make an initial guess for the structure, LD is best with random initial conditions, CD with zero
    initial_guess = np.ones_like(struct) / np.sqrt(3)
    # initialize reconstruction class and perform reconstruction
    recons = Reconstruction(initial_guess, rotations=rot, projections=projections, pol=pol, iterations=it,
                            mask=mask, learning_parameter=learning)
    recons.reconstruct()
    norm_recons = vectorop.normalize(recons.magnetization)

    sinogram_comparison = np.concatenate([recons.sinogram, exp.sinogram], axis=1)
    a = plot_3d(sinogram_comparison, show=False, cmap=cmap, axes_names=(r"$\theta$", "x", "y"), init_take=0)
    b = plot_3d(norm_recons, axial=True if kind == 'LD' else False, arrow_skip=1, show=False)
    c = plot_3d(struct, axial=True if kind == 'LD' else False, arrow_skip=1)


def scalar_tomo(angles):
    """Example of single axis scalar tomography."""
    # get a scalar field to reconstruct
    struct = _get_structure()[0]

    # calculate projections
    projections = radon(struct, angles)
    plot_3d(projections, cmap='Spectral', init_take=0, axes_names=(r"$\theta$", "x", "y"), title="Projections")

    # reconstruct the object from projections
    recons = inv_radon(projections, angles)

    a = plot_3d(struct, cmap='Spectral', init_take=1, init_slice=47, show=False, title="Initial Structure")
    b = plot_3d(recons, cmap='Spectral', init_take=1, init_slice=47, title="Reconstruction")


def _get_mask(shape):
    """Makes a cylinder mask for the provided (vector field) shape."""
    xx, yy, zz = structures.create_mesh(*shape[-3:])
    r = np.sqrt(xx ** 2 + zz ** 2) < shape[1] / 4  # radius mask based on the x size
    h = shape[2] // 4 > np.abs(yy)  # radius mask based on the y size
    return r * h


def _get_structure():
    """Returns an example magnetization vector field from ThreeDViewer package."""
    data_path_resource = resources.files(ThreeDViewer.data) / 'cylinder.ovf'
    with data_path_resource as resource:
        data = io.load_ovf(resource).magnetization

    # make y out-of-plane to match coordinate system
    data = np.array(data[[1, 2, 0]].transpose((0, 2, 3, 1)))
    # mask to leave room for tilting
    return data * _get_mask(data.shape)


def _evenly_spaced(initial, final, steps):
    """Creates an evenly spaced array assuming the ends are cyclic."""
    dx = (final - initial) / steps
    return np.linspace(initial, final - dx, steps)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    initial_angle, final_angle, angle_steps = 0, 360, 18
    rotation_angles = _evenly_spaced(initial_angle, final_angle, angle_steps)

    # example of scalar tomography
    scalar_tomo(rotation_angles)

    # example of vector or orientation tomography
    tilt_angles = [30, -30]
    iteration_number = 10
    learn_param = 30
    dichroism = 'CD'  # circular ('CD') or linear ('LD')
    vector_tomo(rotation_angles, tilt_angles, kind=dichroism, it=iteration_number, learning=learn_param)
