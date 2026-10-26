"""Generic oracles. Nothing here may know about a specific family, module or workflow."""
import jax.numpy as jnp
import numpy as np


def fd_jacobian(f, x, eps=1e-6):
    """Central-difference Jacobian of f at x, shape (f(x).size, x.size)."""
    x = np.asarray(x, dtype=np.float64)

    def at(dx):
        return np.asarray(f(jnp.asarray((x.ravel() + dx).reshape(x.shape)))).ravel()

    return np.stack([(at(e) - at(-e)) / (2 * eps) for e in np.eye(x.size) * eps], axis=1)


def fd_log_volume(f, x, eps=1e-6):
    """log of the factor by which f scales volume at x: 0.5 * log det(J^T J).

    Equals log|det J| for square J; for tall J it is the volume factor onto
    the image subspace.
    """
    J = fd_jacobian(f, x, eps)
    return 0.5 * np.linalg.slogdet(J.T @ J)[1]
