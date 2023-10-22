import numpy as np


def matching_pursuit(y: np.ndarray,
                     D: np.ndarray,
                     eps_min=1e-3,
                     iter_max=1000,
                     verbose=False) -> np.ndarray:
    """

    Args:
        y: time_series_length. Target_vector for sparse representation.
        D: num_basis * time_series_length
        eps_min: eps
        iter_max: max iteration for MP.

    Returns:
        1 * num_basis An amplitude vector x, where y = D * x = x^T * D
    """
    s = np.zeros(D.shape[0], dtype=np.float64)
    r = y.astype(np.float64)
    ii = 0
    for ii in range(iter_max):
        dot = np.dot(D, r)
        i = np.argmax(np.abs(dot))
        s[i] = dot[i]
        r -= s[i] * D[i, :]
        if np.linalg.norm(r) < eps_min:
            break
        if (ii+1)  % 1000 == 0 and verbose:
            print('iter %d pursuit loss=%.6f' % (ii+1, np.linalg.norm(r)))
    if verbose:
        print('iter %d pursuit loss=%.6f' % (ii + 1, np.linalg.norm(r)))
    return np.atleast_2d(s)