"""zero_sum(4): maps x in R^3 onto {y in R^4 : sum(y) = 0} (Householder map).

With r = sqrt(3 + 1) = 2 and s = sum(x):
  forward  y = [x - s/(r(r+1)), -s/r] = [x - s/6, -s/2]
  inverse  x = y[:3] - (y[3] + sum(y)/r)/(r+1) = y[:3] - y[3]/3      (since sum(y) = 0)
  log-det  0: the map is an isometry onto the zero-sum subspace.

KNOWN, by hand:
  x = (1, 1, 1):  s = 3, y = (1 - 1/2, 1 - 1/2, 1 - 1/2, -3/2) = (1, 1, 1, -3)/2
  x = (1, 0, 0):  s = 1, y = (1 - 1/6, -1/6, -1/6, -1/2) = (5, -1, -1, -3)/6
"""
import jax.numpy as jnp
import numpy as np
import pytest

from util import zero_sum

C = zero_sum(4)

KNOWN = [  #            x                     y                  log-det
    pytest.param([1.0, 1.0, 1.0], [1/2, 1/2, 1/2, -3/2],   0.0, id="x=(1,1,1)->y=(1,1,1,-3)/2"),
    pytest.param([1.0, 0.0, 0.0], [5/6, -1/6, -1/6, -1/2], 0.0, id="x=(1,0,0)->y=(5,-1,-1,-3)/6"),
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