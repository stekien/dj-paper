import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from typing import Mapping, Protocol


def ravelize_function(f, pytree):
    """
    Takes a function that accepts a PyTree and a PyTree,
    and produces a function that accepts a flat array.
    """
    # note: ravel_pytree is only really safe when we
    # know all the dtypes are the same. See
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.flatten_util.ravel_pytree.html
    # This is usually true in stats models
    _, unravel = ravel_pytree(pytree)
    return lambda x: f(unravel(x))


class Shaped(Protocol):
    shape: tuple[int, ...]
    dtype: jnp.dtype


class ParameterConstraint:
    def __init__(self, shape=(), dtype=jnp.float32):
        # if any(s < 0 for s in shape):
        # raise ValueError("Shape dimensions must be non-negative")

        self.shape = shape
        self.dtype = dtype

    def __call__(self, x):
        return x

    def inverse(self, y):
        return y

    def jacobian(self, _):
        return 0.0


class bind:
    """Constraint built from earlier parameters: bind("a", "b", lambda a, b: box(...))."""
    def __init__(self, *args):
        *parents, self.f = args
        self.parents = tuple(parents)
        self.depends_on = frozenset(self.parents)

    def make(self, theta):
        return self.f(*(theta[p] for p in self.parents))


# simple alias: base class does transforms
real = ParameterConstraint


# basic example of a positive constraint
class positive(ParameterConstraint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        return jnp.exp(x)

    def inverse(self, y):
        return jnp.log(y)

    def jacobian(self, x):
        return jnp.sum(x)


# basic example of a box constraint
class box(ParameterConstraint):
    def __init__(self, lower=-jnp.inf, upper=jnp.inf, **kwargs):
        self.lower = lower
        self.upper = upper
        super().__init__(**kwargs)

    def __call__(self, x):
        return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(x)

    def inverse(self, y):
        return jnp.log((y - self.lower)) - jnp.log((self.upper - y))

    def jacobian(self, x):
        return jnp.sum(jnp.log(self.upper - self.lower)
                - jax.nn.softplus(x)
                - jax.nn.softplus(-x)
               )


# note: shape will need some care for something like a simplex,
# which should also include axis=... in its definition. The shape
# should be that of the unconstrained parameter.
class simplex(ParameterConstraint):
    def __init__(self, shape, axes=..., **kwargs):
        self.axes = axes
        shape_n = jnp.array(shape, dtype=int)
        shape_n = shape_n.at[axes,].set(shape_n[axes,] - 1)

        super().__init__(shape=tuple(shape_n), **kwargs)

    def __call__(self, x):
        raise NotImplementedError("Simplex constraint not yet implemented")

    def inverse(self, y):
        raise NotImplementedError("Simplex constraint not yet implemented")

    def jacobian(self, x):
        raise NotImplementedError("Simplex constraint not yet implemented")


def spec_to_pytree(
    parameter_spec: Mapping[str, Shaped | bind],
) -> dict[str, jnp.ndarray]:
    """
    Turns a dictionary from parameter names to shapes/dtypes
    into a pytree of zeros with those shapes/dtypes.

    Entries can be anything satisfying the Shaped protocol (ParameterConstraint
    objects, jax.numpy arrays) or a bind, whose shape comes from the
    constraint it builds.
    """
    out = {}
    def get(name):
        if name not in out:
            v = parameter_spec[name]
            if isinstance(v, bind):
                v = v.make({p: get(p) for p in v.parents})
            out[name] = jnp.zeros(v.shape, dtype=v.dtype)
        return out[name]
    return {k: get(k) for k in parameter_spec}


def constrain(parameter_spec: Mapping[str, ParameterConstraint], **kwargs):
    """
    Constrain parameters per the spec and compute the log Jacobian.
    Anything missing from the spec is unconstrained. Binds resolve their
    parents first (at trace time, so no cost under jit).
    """
    parameters = {}
    jacobian = 0.0

    def get(name):
        nonlocal jacobian
        if name not in parameters:
            c = parameter_spec.get(name, real())
            if isinstance(c, bind):
                c = c.make({p: get(p) for p in c.parents})
            parameters[name] = c(kwargs[name])
            jacobian += c.jacobian(kwargs[name])
        return parameters[name]
    
    # get() constrains a bind's parents first (recursively), and skips
    # anything already done, so each parameter and its Jacobian term is
    # computed exactly once, whatever the order of kwargs.
    for name in kwargs:
        get(name)
    return parameters, jacobian


def unconstrain(parameter_spec: Mapping[str, ParameterConstraint], **kwargs):
    """Inverse of constrain. Anything missing from the spec is unconstrained."""
    parameters = {}
    for name, y in kwargs.items():
        c = parameter_spec.get(name, real())
        if isinstance(c, bind):
            c = c.make(kwargs)  # parents are already constrained here
        parameters[name] = c.inverse(y)
    return parameters


# version that assumes data is closed over in
# the passed-in functions.
# Could easily change to pass data later
def make_log_density(
    log_prior,
    log_likelihood,
    parameter_spec: Mapping[str, ParameterConstraint] = dict(),
):
    """
    Make a log_density function from a log_prior, log_likelihood,
    and (optionally) a function to constrain parameters.

    Parameters
    ----------
    log_prior : function
        This function will be passed the parameters
        by name, and should return the log of the prior density.
    log_likelihood : function.
        This function will be passed the parameters
        by name, and should return the log of the likelihood.
    parameter_spec : dict
        A dictionary mapping parameter names to ParameterConstraint
        objects.

    Returns
    -------
    function
        A function that computes the log density of the model.
    """

    @jax.jit
    def log_density(unc_params):
        params, log_det_jac = constrain(parameter_spec, **unc_params)
        return log_det_jac + log_prior(**params) + log_likelihood(**params)

    return log_density


# similar to a solution found at https://github.com/jax-ml/jax/discussions/9508#discussioncomment-2144076,
# but uses ravel_pytree to avoid needing to split the key
def init_random(parameter_spec, rng_key, radius=2):
    """
    Given a tree and a random key, return a tree with the same structure
    but with each leaf replaced by a random uniform value in the range [-radius, radius].
    """
    d, unravel = jax.flatten_util.ravel_pytree(spec_to_pytree(parameter_spec))
    uniforms = jax.random.uniform(rng_key, shape=d.shape, minval=-radius, maxval=radius)
    return unravel(uniforms)


def init(parameter_spec, parameters):
    return unconstrain(parameter_spec, **parameters)
