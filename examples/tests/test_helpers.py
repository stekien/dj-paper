"""fd_jacobian and fd_log_volume, checked on maps with exactly known Jacobians."""
import jax.numpy as jnp
import numpy as np

from helpers import fd_jacobian, fd_log_volume

# R^3 -> R^4 with J = TALL everywhere; J^T J = I + 1 1^T has eigenvalues 4, 1, 1
TALL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])


def test_fd_jacobian_of_linear_map__is_its_matrix():
    J = fd_jacobian(lambda x: jnp.asarray(TALL) @ x, jnp.ones(3))
    np.testing.assert_allclose(J, TALL, atol=1e-6)


def test_fd_jacobian_of_elementwise_exp__is_diagonal_of_exp_x():
    # x = (0, log 2, log 3): d exp(x_i) / d x_j = exp(x_i) if i == j, else 0
    x = jnp.log(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(fd_jacobian(jnp.exp, x), np.diag([1.0, 2.0, 3.0]), atol=1e-6)


def test_fd_log_volume_of_square_map__is_log_abs_det():
    # J = diag(-1, 2, 3): |det J| = 6; the negative entry checks the absolute value
    D = jnp.diag(jnp.array([-1.0, 2.0, 3.0]))
    np.testing.assert_allclose(fd_log_volume(lambda x: D @ x, jnp.ones(3)), np.log(6.0), atol=1e-6)


def test_fd_log_volume_of_tall_map__is_half_log_det_of_gram():
    # det(J^T J) = det(I + 1 1^T) = 4 * 1 * 1, so 0.5 * log 4 = log 2
    f = lambda x: jnp.asarray(TALL) @ x
    np.testing.assert_allclose(fd_log_volume(f, jnp.ones(3)), np.log(2.0), atol=1e-6)