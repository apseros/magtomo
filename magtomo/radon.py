import numpy as np
import scipy
import magpack.rotations as rtn


def radon(image, thetas=None, order=1):
    """Performs the radon transform on an image.

    Summation / integration occurs along the last dimension. Therefore, an input image with shape (x, y) and (n,) angles
    will result in an output image with shape (n, x). For 3D input of shape (x, y, z) and (n,) the output image will
    have the shape (n, x, y).

    Parameters
    ----------
    image : np.ndarray
        The image to transform.
    thetas : np.ndarray | list (optional)
        Angles to calculate the radon transform.
    order : int (optional)
        Interpolation order.

    Returns
    -------
    np.ndarray
        The Radon transform of the image, the 0th index represents the measured angles.
    """
    thetas = np.asarray(thetas)

    # first index is the number of angles
    sinogram = np.zeros((thetas.shape[0], *image.shape[:-1]))
    for (ii, theta) in enumerate(thetas):
        if not isinstance(theta, np.ndarray):
            theta = rtn.rot(theta) if image.ndim == 2 else rtn.roty(theta)
        sinogram[ii, ...] = rtn.rotate_scalar_field(image, theta, order=order).sum(axis=-1)
    return sinogram


def _get_fourier_filter(size, filter_name):
    """Returns the Fourier filter of specified size (excerpt from scikit-image). """
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(np.fft.fft(f))  # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * np.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = np.fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= np.fft.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= np.fft.fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]


def inv_radon(radon_image, theta=None, filter_name="ramp"):
    """Computes the inverse radon transform of a sinogram / stack of projections.

    Parameters
    ----------
    radon_image : np.ndarray
        The sinogram / stack of projections.
    theta : np.ndarray (optional)
        Angles to calculate the radon transform.
    filter_name : str (optional)
        The name of the filter to use.
    Returns
    -------
    np.ndarray
        The inverse radon transform of the sinogram / stack of projections."""
    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[0], endpoint=False)
    expanded = False
    parity = 1
    if radon_image.ndim == 2:
        radon_image = radon_image[:, :, np.newaxis]  # odd permutation, changes axes handedness!
        parity = -1
        expanded = True

    if radon_image.ndim != 3:
        raise ValueError("Radon image must be 2D or 3D.")

    angles_count = len(theta)
    if angles_count != radon_image.shape[0]:
        raise ValueError("The given angles do not match the number of projections.")

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter_name not in filter_types:
        raise ValueError(f"Unknown filter: {filter_name}")

    img_size = radon_image.shape[1]
    projection_size_padded = np.max((64, np.asarray(2 ** np.ceil(np.log2(img_size)), dtype=int)))

    fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)[np.newaxis, ...]
    pad_width = np.asarray([[0, 0], [0, projection_size_padded - img_size], [0, 0]])

    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    projection = np.fft.fft(img, axis=1) * fourier_filter
    radon_filtered = np.real(np.fft.ifft(projection, axis=1))[:, :img_size, ...]
    # Reconstruct image by interpolation
    reconstructed = np.zeros((*radon_image.shape[1:], img_size))
    xpr, zpr = np.mgrid[:img_size, :img_size] - img_size // 2
    x = np.arange(img_size) - img_size // 2

    for plane, angle in zip(radon_filtered, np.deg2rad(theta)):
        t = xpr * np.cos(angle) + parity * zpr * np.sin(angle)
        line_interp = scipy.interpolate.interp1d(x, plane, kind='linear', bounds_error=False, fill_value=0, axis=0)
        reconstructed += line_interp(t).transpose((0, 2, 1))

    final_recons = reconstructed * np.pi / (2 * angles_count)
    final_recons = final_recons[:, 0, :] if expanded else final_recons
    return final_recons

