"""Contract of the constraints family: every invariant holds for every entry in CASES."""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import fd_log_volume
from util import positive, zero_sum

class Case(NamedTuple):
    make: Callable                   # () -> constraint, 3 to 5 elements per axis
    in_support: Callable             # (c, y) -> True iff y is finite and satisfies the constraint
    free: Callable = lambda c, y: y  # (c, y) -> y without outputs determined by the others


def is_positive(c, y):
    return np.isfinite(y).all() and (y > 0).all()


def is_zero_sum(c, y):
    sums = [y.sum(axis=a) for a in c.axes]
    return np.isfinite(y).all() and all(np.allclose(s, 0.0, atol=1e-10) for s in sums)


CASES = {
    "positive":    Case(make=lambda: positive(shape=(3,)), in_support=is_positive),
    "zero_sum_1d": Case(make=lambda: zero_sum(5), in_support=is_zero_sum),
    "zero_sum_2d": Case(make=lambda: zero_sum((3, 4), axes=(0, 1)), in_support=is_zero_sum),
    "zero_sum_3d": Case(make=lambda: zero_sum((3, 4, 5), axes=(0, 1, 2)), in_support=is_zero_sum),
    "zero_sum_4d": Case(make=lambda: zero_sum((3, 4, 3, 3), axes=(0, 1, 2, 3)), in_support=is_zero_sum),
}

OUTPUTS = {"forward": lambda c, x: c(x), "log_det": lambda c, x: c.jacobian(x)}

each_case = pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
each_output = pytest.mark.parametrize("output", list(OUTPUTS.values()), ids=list(OUTPUTS))


def random_x(c, batch=()):
    """Fixed-seed standard normal draw of the unconstrained shape, optionally batched."""
    return jnp.asarray(np.random.default_rng(0).normal(size=batch + c.shape))


@each_case
def test_inverse_after_forward__recovers_x(case):
    c = case.make()
    x = random_x(c)
    np.testing.assert_allclose(c.inverse(c(x)), x, atol=1e-10)


@each_case
def test_forward_output__lies_in_support(case):
    c = case.make()
    assert case.in_support(c, np.asarray(c(random_x(c))))


@each_case
def test_log_det__matches_finite_differences(case):
    c = case.make()
    x = random_x(c)
    expected = fd_log_volume(lambda u: case.free(c, c(u)), x)
    np.testing.assert_allclose(c.jacobian(x), expected, atol=1e-6)


@each_output
@each_case
def test_output_under_jit__equals_eager_output(case, output):
    c = case.make()
    x = random_x(c)
    np.testing.assert_allclose(jax.jit(lambda u: output(c, u))(x), output(c, x), atol=1e-10)


@each_output
@each_case
def test_output_under_vmap__equals_output_per_item(case, output):
    c = case.make()
    xs = random_x(c, batch=(3,))
    per_item = np.stack([output(c, x) for x in xs])
    np.testing.assert_allclose(jax.vmap(lambda u: output(c, u))(xs), per_item, atol=1e-10)