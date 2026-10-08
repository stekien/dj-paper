from jax import random, jit, lax
import jax
import jax.numpy as jnp
from jax.scipy import stats

from util import (
    bind,
    ravelize_function,
    make_log_density,
    constrain,
    positive,
    real,
    box,
    spec_to_pytree,
)

# These are the primary exports of this module:
__all__ = [
    "log_density",
    "log_density_vec",
    "constraints",
    "generated_quantities",
    "generated_quantities_vec",
]

# ---------- simulate data from a garch(1,1) ----------
rng_key = random.key(0)
data_key, rng_key = random.split(rng_key)
true = {"alpha_0": 0.1, "alpha_1": 0.1, "beta_1": 0.85}
T = 1000

def simulate(key, alpha_0, alpha_1, beta_1, T):
    def step(s2, e):
        r = jnp.sqrt(s2) * e
        return alpha_0 + alpha_1 * r**2 + beta_1 * s2, (r, s2)
    _, (r, s2) = lax.scan(step, alpha_0 / (1 - alpha_1 - beta_1), random.normal(key, (T,)))
    return r, s2

returns, true_s2 = simulate(data_key, **true, T=T)

## Parameter definitions

# This can be partial (or even entirely omitted) if you do not require any reshaping utilities
#   (ravelize_function, spec_to_pytree, init_random, etc.),
# otherwise it must cover all parameters to get the shapes/dtypes correct.
parameter_spec = {
    "alpha_0": positive(),
    "alpha_1": box(lower=0.0, upper=1.0),
    "beta_1": bind("alpha_1", box, lower=0.0, upper=lambda a: (1 - a)),
}


# log density components


def log_prior(alpha_0, alpha_1, beta_1):
    lp_alpha_0 = jnp.sum(stats.norm.logpdf(alpha_0, loc=0.0, scale=1.0))
    lp_alpha_1 = 0.0 # flat prior
    lp_beta_1 = 0.0 # flat prior
    return lp_alpha_0 + lp_alpha_1 + lp_beta_1


def log_likelihood(alpha_0, alpha_1, beta_1):
    r = returns
    def step(s2, r_prev):
        return alpha_0 + alpha_1 * r_prev**2 + beta_1 * s2, s2
    _, s2 = jax.lax.scan(step, alpha_0 / (1 - alpha_1 - beta_1), r)
    return jnp.sum(stats.norm.logpdf(r, loc=0.0, scale=jnp.sqrt(s2)))


# a log density function
log_density = make_log_density(log_prior, log_likelihood, parameter_spec=parameter_spec)


# We can also provide a flattened version, automatically,
# using the structure of the parameters defined above.
log_density_vec = ravelize_function(log_density, spec_to_pytree(parameter_spec))


# we might also want something like "generated quantities"
@jit
def generated_quantities(rng, r_new, **params):
    constrained, _ = constrain(parameter_spec, **params)
    alpha_0, alpha_1, beta_1 = constrained["alpha_0"], constrained["alpha_1"], constrained["beta_1"]
    r = returns
    def step(s2, r_prev):
        return alpha_0 + alpha_1 * r_prev**2 + beta_1 * s2, s2
    s2_curr, _ = jax.lax.scan(step, alpha_0 / (1 - alpha_1 - beta_1), r)
    sigma_new = jnp.sqrt(alpha_0 + alpha_1 * r_new**2 + beta_1 * s2_curr)
    return {"alpha_0": alpha_0, "alpha_1": alpha_1, "beta_1": beta_1,
             "sigma_new": sigma_new}


# and a flattened version
@jit
def generated_quantities_vec(rng, r_new, params_vec):
    gq = lambda param_dict: generated_quantities(rng, r_new, **param_dict)
    return ravelize_function(gq, spec_to_pytree(parameter_spec))(params_vec)