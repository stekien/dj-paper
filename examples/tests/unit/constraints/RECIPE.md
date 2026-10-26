# Adding a constraint

## Contract

A constraint subclasses `ParameterConstraint` and implements `__call__`
(forward), `inverse`, and `jacobian` (log|det J| of the forward map). `c.shape`
is the unconstrained shape. Every entry in `CASES` passes every invariant
below; there are no per-entry exclusions. `x` is a fixed-seed standard normal
draw.

- `test_inverse_after_forward__recovers_x`: inverse(forward(x)) == x.
- `test_forward_output__lies_in_support`: forward(x) is finite and satisfies the constraint.
- `test_log_det__matches_finite_differences`: jacobian(x) == fd_log_volume of x -> free(forward(x)).
- `test_output_under_jit__equals_eager_output`: for forward and log-det.
- `test_output_under_vmap__equals_output_per_item`: for forward and log-det, batch of 3.

## Registry entry

Add to `CASES` in `test_properties.py`, keyed `<member>`, or
`<member>_<configuration>` when the member has several entries
(e.g. `zero_sum_1d`, `zero_sum_2d`):

    "<member>": Case(make=lambda: <member>(<args>), in_support=<is_support>),

- `make`: builds the constraint with 3 to 5 elements per axis.
- `in_support`: a function `is_<support>(c, y)` defined next to `is_positive`
  and `is_zero_sum`, returning True iff `y` is finite and satisfies the
  constraint. Reuse an existing one if it fits.
- `free`: omit unless some outputs are determined by the others. Then pass a
  function returning `y` without them (e.g. simplex: drop the last entry along
  `c.axis`, since it equals 1 minus the rest). `jacobian` is defined with
  respect to the remaining coordinates.

A second entry for the same member is justified only by a genuinely different
configuration: array-valued arguments, another axis, several axes. Never by a
different input value.

## Member file

Copy `test_positive.py` to `test_<member>.py`; 30 to 45 lines.

- Module docstring: the forward, inverse and log-det formulas for the
  configuration in the file, then each `KNOWN` row derived by hand in one or
  two lines.
- `C = <member>(...)`: one module-level instance with 3 to 5 elements.
- `KNOWN`: 2 or 3 rows `pytest.param(x, y, log_det, id="x=...->y=...")`.
  Inputs with short, exact arithmetic (0, log 2, log 3, midpoints, vectors of
  ones), |x| <= 10.
- Exactly these tests, each parametrized over `KNOWN`, plus regression tests:
  - `test_forward_at_x__gives_hand_computed_y`
  - `test_inverse_at_y__gives_hand_computed_x`
  - `test_log_det_at_x__gives_hand_computed_value`

## Integration

Only if the member's constructor takes arguments that can come from other
parameters through `bind` (e.g. box bounds), and
`tests/integration/test_log_density.py` has no test for the member yet: add
one test function there, named
`test_log_density_with_<prior>_prior_bounded_by_bind__integrates_to_one`.
Copy `test_log_density_with_uniform_prior_bounded_by_bind__integrates_to_one`
(box).

- Spec, defined inside the test: a parent with a normalized prior, and the
  member bound to it through `bind`, with a normalized conditional prior on
  the member's support.
- Comment: the log density derived by hand, showing how the member's
  log-Jacobian combines with its conditional prior, and a bound on the
  integrand outside the grid.
- Assertion: `exp(log_density)` summed over a grid of both unconstrained
  values that covers the tails equals 1 at `rtol=1e-6`.

Otherwise, no integration test.

## Files touched

- `tests/unit/constraints/test_properties.py`: the new `CASES` entries and, if
  needed, one `is_<support>` function.
- `tests/unit/constraints/test_<member>.py`: new.
- `tests/integration/test_log_density.py`: one test, only if "Integration"
  applies.

Anything else is out of scope.