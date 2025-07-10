from scipy.special import logsumexp
import numpy as np
import random


def inverse_transform_sampler(logpmf, Nsamples=1):
    """Function to perform inverse transform sampling on the input discrete log
        probability density values

    Args:
        logpmf (np.ndarray): discrete log probability density values
        Nsamples (int, optional): Number of wanted samples. Defaults to 1.

    Returns:
        np.ndarray: Sampled indices of the input axis
    """
    logpmf = logpmf - logsumexp(logpmf)
    logcdf = np.logaddexp.accumulate(logpmf)
    cdf = np.exp(logcdf-logcdf[-1])  

    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(cdf, u) for u in randvals]
    return indices

# Need to figure out a more rigorous solution. Stable for up to ...
def integral_inverse_transform_sampler(logpmf, axes=None, Nsamples: int = 1):

    Nsamples = int(round(Nsamples))

    logpmf_flattened = logpmf.flatten()

    logpmf_flattened = logpmf_flattened - logsumexp(logpmf_flattened)

    flat_logcdf = np.logaddexp.accumulate(logpmf_flattened)

    flat_cdf = np.exp(flat_logcdf-flat_logcdf[-1])  

    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(flat_cdf, u) for u in randvals]

    reshaped_simulated_indices = np.unravel_index(indices, np.squeeze(logpmf).shape)

    return [axis[index] for axis, index in zip(axes, reshaped_simulated_indices)]




def vectorised_inverse_transform_sampler(logpmfs, axes=None, Nsamples: list[int] = [1], reshape_shape=None):

    Nsamples = [int(round(num)) for num in Nsamples]

    logpmfs_flattened = logpmfs.reshape(logpmfs.shape[0], -1)
    logpmfs_flattened = logpmfs_flattened - logsumexp(logpmfs_flattened, axis=1)[:, None]
    flat_logcdfs = np.logaddexp.accumulate(logpmfs_flattened, axis=1)
    flat_cdfs = np.exp(flat_logcdfs-flat_logcdfs[:, -1][:, None])
    # Vectorized random value generation, not sure whether it's any better than list comprehension
    randvals_list = np.random.random((len(Nsamples), max(Nsamples)))

    indices_list = [np.searchsorted(flat_cdf, randvals[:num]) for flat_cdf, randvals, num in zip(flat_cdfs, randvals_list, Nsamples)]

    samples  = [[] for axis in axes]

    for indices in indices_list:

        reshaped_simulated_indices = np.unravel_index(indices, np.squeeze(logpmfs[0, :]).shape)

        [samples[axis_index].extend(axis[index]) for axis_index, (axis, index) in enumerate(zip(axes, reshaped_simulated_indices))]

    return samples
    