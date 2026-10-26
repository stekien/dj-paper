# Testing

Read this before touching `tests/`. It fixes where every kind of test goes, so
the tree grows the same way no matter who adds to it. Keep every change
minimal: the fewest lines that test the claim. Copy the style of the existing
files exactly.

## Layout

    tests/
      TESTING.md                  this file
      conftest.py                 global settings only (e.g. float64); no fixtures
      helpers.py                  generic oracles: finite differences, ...
      test_helpers.py             tests of the helpers against analytic cases
      unit/
        <family>/                 one folder per family
          RECIPE.md               how to add a member to this family
          test_properties.py      Case, CASES registry, invariants on every entry
          test_<member>.py        hand-computed values for one member
        <path>/test_<module>.py   one file per non-family module, mirroring src
      integration/
        test_<workflow>.py        one file per user workflow

Run: `pytest tests -m "not slow"` while working, `pytest tests` before finishing.
The pytest configuration (import mode, markers) lives in `pyproject.toml`.

## Kinds of code

All public code is exactly one of three kinds. The kind decides where its
tests live.

A **family** is two or more implementations of one interface that must satisfy
the same invariants. Its tests split in two. The contract is checked once, for
every member, through the `CASES` registry in `test_properties.py`. The values
specific to one member are pinned in `test_<member>.py`. The family's
`RECIPE.md` fixes the registry fields, the member-file template and its exact
list of tests. The folder is named after the family, even if all members live
in one source module.

A **module** is any other public code. Its tests live at the path that mirrors
the source tree: the tests for `<package>/a/b.py` are in
`tests/unit/a/test_b.py`, and those for a top-level `util.py` in
`tests/unit/test_util.py`. Test public behaviour only; private functions are
covered through the public ones.

A **workflow** is a path through the public API that combines components the
way a user would. Its tests live in `integration/test_<workflow>.py`. A claim
about how components interact is an integration test. A claim about one
component is a unit test, even if building that component needs others.

## How the tree grows

Find the row that matches your change. Touch only the files in that row.

| Your change adds | Touch only |
|---|---|
| a member of an existing family | its entry in `unit/<family>/test_properties.py`; a new `unit/<family>/test_<member>.py`; one integration test if `RECIPE.md` requires it |
| an invariant every member must satisfy | one test in `unit/<family>/test_properties.py` |
| a new family | a new `unit/<family>/` with `RECIPE.md`, `test_properties.py`, and the moved tests of the first member |
| public behaviour in a module | `unit/<path>/test_<module>.py`, created if missing |
| a new user workflow | a new `integration/test_<workflow>.py` |
| a new case of an existing workflow | one test function in that workflow's file, with its own small spec defined inside the test |
| a bug fix | one regression test in the file that owns the claim |
| a shared helper | `helpers.py` and `test_helpers.py` |

A module becomes a family when a second implementation of its interface
appears. In that change, move its existing tests into
`unit/<family>/test_<first member>.py` without editing them. Bring that file
in line with the recipe in a separate change.

A new invariant runs on every existing entry. If one fails, follow "When a
test fails". Never exclude the entry.

## Family recipe template

Every `unit/<family>/RECIPE.md` has these sections, in this order, and
nothing else:

    # Adding a <member>

    ## Contract
    The invariants in test_properties.py, one line each. Every entry in CASES
    passes all of them; there are no per-entry exclusions. Where they apply,
    the contract includes compatibility with the
    transformations the package supports (e.g. jit, vmap).

    ## Registry entry
    The fields of Case and what each must do, the shape to build with, and
    when a second entry for the same member is justified (a genuinely
    different configuration, never a different input value).

    ## Member file
    The reference file to copy, the length budget, what the module docstring
    contains (formulas and hand derivation), the KNOWN table, and the exact
    list of tests, names included.

    ## Integration
    The condition under which a member also needs an integration test, and
    which workflow file it goes in.

    ## Files touched
    The complete list. Anything else is out of scope.

## Rules

### Names

- `test_<what is examined>__<expected outcome>`, read aloud as a falsifiable
  claim. Good: `test_inverse_at_y__gives_hand_computed_x`,
  `test_bind_with_unknown_name__raises_keyerror`.
  Bad: `test_box`, `test_roundtrip`.
- Parametrize ids state the evidence, without spaces: `id="x=log2->y=2"`.
- One claim per test. If the name needs "and", split the test. Several
  asserts are fine only when they check the same claim.

### Expected values

Every expected value comes from exactly one of these sources:

- a hand derivation written next to it;
- a helper from `tests/helpers.py`;
- an analytic identity (e.g. a density integrates to 1,
  `inverse(forward(x)) == x`);
- for invariance under a transformation the package supports (jit, vmap), the
  same computation without the transformation;
- for errors, the documented contract:
  `pytest.raises(<Error>, match=...)` with the exception type and message the
  documentation promises.

Never compute it with the code under test or anything in the package it
depends on (except as the untransformed side of a transformation invariant),
and never paste the output of an earlier run. Autodiff may be the thing under
test (compared against finite differences), never the oracle.

### Inputs

- For hand-computed values, choose inputs where the arithmetic is short and
  exact: 0, log 2, log 3, midpoints, vectors of ones. No |x| > 10. Small
  shapes: 3 to 5 elements.
- Extreme inputs appear only in contract invariants that check outputs are
  finite and valid, and in the grids of numerical integrals, which must cover
  the tails. Never in hand-computed values.
- Random inputs only through `np.random.default_rng(0)`.

### Tolerances

- `atol=1e-10` for exact values.
- `atol=1e-6` against finite differences.
- `rtol=1e-6` for numerical integrals.

Never loosen them.

### Structure

- No test classes, fixtures, skip or xfail. Plain data types such as `Case`
  are fine.
- Only these markers and skips are allowed:
  - `@pytest.mark.parametrize`;
  - `pytest.importorskip(...)` at the top of a file that needs an optional
    dependency;
  - `@pytest.mark.slow` (registered in the pytest config) for a test that
    takes more than a few seconds.
- Pytest imports test files with `--import-mode=importlib`, so names like
  `test_properties.py` may repeat across folders. Test files never import
  each other.
- Shared code lives only in `tests/helpers.py`. Use what is there instead of
  writing your own. Helpers stay generic: anything that knows about a
  specific family, module or workflow belongs in that test file, or in the
  family's `test_properties.py`.
- Helpers are additive. Add new ones; never change the signature or behaviour
  of an existing one. Every new helper gets a test in `test_helpers.py`
  against a case with a known analytic answer.
- Inputs shared by several tests in a file are module-level constants
  (e.g. `C`, `SPEC`, `U`). Existing hand values depend on them: never change
  their values. Define a new constant, or a small spec inside the test. A
  `CASES` registry only grows by new entries.
- Member files contain exactly the tests their recipe lists, plus regression
  tests.

### Regression tests

A fixed bug gets one test in the file that owns the claim. Its docstring cites
the issue and the input that triggered the bug. Its expected value follows the
rules above like any other.

### Changing existing tests

By default, don't. There are two exceptions:

- **An intentional behaviour change.** Update the affected expected values and
  their derivations in the same change. List each changed value, with the
  reason, in the change description.
- **Creating a family.** Move the existing tests unchanged.

Never loosen a tolerance, replace a hand value with a computed one, or delete
a failing test.

### When a test fails

Do not change the expected value or tolerance. Re-check the derivation. If it
holds, stop and report the failing numbers: it is a bug.

## Done when

- `pytest tests` passes.
- Every new expected value has its source next to it: a derivation, a helper,
  an identity, or a documented error.
- The diff touches only the files in your row of "How the tree grows".
- A new family has `RECIPE.md` and `test_properties.py`.
- A new helper has a test in `test_helpers.py`.