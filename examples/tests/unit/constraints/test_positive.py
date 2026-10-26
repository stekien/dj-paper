"""positive: y = exp(x), x = log(y), log|det J| = sum(x).

KNOWN, by hand:
  x = (0, log 2, log 3):        y = (1, 2, 3), log-det = 0 + log 2 + log 3 = log 6
  x = (log 2, log 2, log 2):    y = (2, 2, 2), log-det = 3 log 2 = log 8
"""
import jax.numpy as jnp
import numpy as np
import pytest

from util import positive

C = positive(shape=(3,))
L2, L3 = np.log(2.0), np.log(3.0)

KNOWN = [  #           x                y              log-det
    pytest.param([0.0, L2, L3], [1.0, 2.0, 3.0], np.log(6.0), id="x=log(1,2,3)->y=(1,2,3)"),
    pytest.param([L2, L2, L2],  [2.0, 2.0, 2.0], np.log(8.0), id="x=log(2,2,2)->y=(2,2,2)"),
]


@pytest.mark.parametrize("x, y, log_det", KNOWN)
def test_forward_at_x__gives_hand_computed_y(x, y, log_det):
    np.testing.assert_allclose(C(jnp.array(x)), y, atol=1e-10)


@pytest.mark.parametrize("x, y, log_det", KNOWN)
def test_inverse_at_y__gives_hand_computed_x(x, y, log_det):
    np.testing.assert_allclose(C.inverse(jnp.array(y)), x, atol=1e-10)


@pytest.mark.parametrize("x, y, log_det", KNOWN)
def test_log_det_at_x__gives_hand_computed_value(x, y, log_det):
    np.testing.assert_allclose(C.jacobian(jnp.array(x)), log_det, atol=1e-10)