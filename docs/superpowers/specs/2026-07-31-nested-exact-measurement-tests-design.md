# Exact Nested Measurement Test Design

## Status

Approved design amendment for Tasks 4 and 5 of the Grassmann nested tensor
network implementation.

## Scope

The production measurement interfaces remain environment-based:

- `compute_nested_exp_site(..., env::CTMRGEnv, ...)`
- `compute_nested_exp_hbond(..., env::CTMRGEnv, ...)`
- `compute_nested_exp_vbond(..., env::CTMRGEnv, ...)`

Their construction continues to reuse the existing CTMRG measurement
primitives. This amendment changes only the Task 4 and Task 5 validation
strategy: those tests must not construct a `CTMRGEnv` or call
`run_GCTMRG!`.

The purpose of these tests is to validate the nested factorization and its
fermionic signs independently of CTMRG convergence.

## Chosen Approach

Use exact finite-periodic contractions implemented with the existing native
`contract` and `trace` operations. For every observable, construct a nested
impurity network and the corresponding reduced-layer reference from the same
PEPS, close both exactly, and compare:

1. the denominator,
2. the unnormalized numerator, and
3. the normalized expectation value.

Comparing all three prevents a common normalization error from being hidden
by division. Dense-array enumeration is not used because it bypasses native
Grassmann contraction semantics. An uniterated CTMRG environment is not used
because its result depends on a random, unconverged boundary.

## Shared Exact-Closure Rules

- Reuse the tested native periodic closure logic from
  `test/nested_network.jl`.
- Cover all four periodic spin structures:
  `(false, false)`, `(true, false)`, `(false, true)`, and `(true, true)`.
- Use seeded deterministic PEPS fixtures and both `Float64` and
  `ComplexF64` where practical.
- Use a nondegenerate fixture for number and bond operators. At least one
  tested numerator must have magnitude above the comparison absolute
  tolerance, so a zero-equals-zero result cannot be the sole evidence.
- Exact nested/reduced comparisons use `rtol=5e-12` and `atol=1e-12` unless
  a tighter bitwise equality is naturally available.
- No test in `test/nested_measurements.jl` may construct `CTMRGEnv`, call
  `initialize_nested_environment`, `run_GCTMRG!`, or `run_nested_GCTMRG!`.

## One-Site Test Flow

Use a small periodic source cell with more than one physical site so the
number observable is nondegenerate.

For a source site and one-site operator:

1. Build the corrected bulk `NestedNetwork`.
2. Copy its rank-four checkerboard matrix.
3. Replace only the source Y site with
   `nested_y_operator(nested, peps, source, operator)`.
4. Build the reduced bulk matrix with `reduced_tensor.(peps.A)`.
5. Copy that matrix and replace only the same source site with
   `reduced_tensor(peps.A[source], operator)`.
6. Close the nested bulk, nested impurity, reduced bulk, and reduced
   impurity exactly under the same spin structure.
7. Compare both denominators, both numerators, and their ratios.

Identity coverage additionally verifies that an identity-dressed Y equals
the bulk Y tensor. Number-operator coverage verifies a nonzero numerator.
Operator dimension, parity split, arrow, source-bound, and matrix-unit-cell
validation remain covered without an environment.

The environment-based scalar and matrix production methods remain in
`algorithms/Nested_CTMRG/measurements.jl`; their dispatch and data-flow
review is deferred to the later integration/acceptance stage.

## Two-Site Test Flow

The production `_operator_schmidt` decomposition remains the shared pure
operator factorization. Its reconstruction is tested exactly and each term
must have a definite one-site parity; odd contributions occur only in
odd-odd pairs.

For a horizontal bond:

1. Use a periodic `1 x 2` source cell.
2. For every operator-Schmidt term, copy the corrected nested checkerboard
   and replace the two endpoint Y tensors with the corresponding
   operator-dressed Ys.
3. Close each term exactly and sum the resulting scalars.
4. Build the reduced reference with
   `reduced_tensor_hbond(A_left, A_right, operator)`.
5. Close the rank-six reduced bond tensor by tracing its external
   horizontal pair and the two vertical pairs with the matching spin
   structure.

For a vertical bond, use a periodic `2 x 1` source cell and the analogous
flow with `reduced_tensor_vbond`; close the two horizontal pairs and the
external vertical pair.

Build denominators with the two-site identity and compare raw denominators,
raw numerators, and normalized horizontal/vertical expectations. The
nested endpoint construction is factored into pure internal helpers so the
same terms are consumed by both the exact tests and the environment-based
public measurement methods.

## Error Handling

The existing explicit errors remain part of the tests:

- source outside the unit cell: `ArgumentError`;
- physical dimensions or parity split mismatch: `DimensionMismatch`;
- operator arrows mismatch: `ArgumentError`;
- operator-matrix unit cell mismatch: `DimensionMismatch`;
- invalid two-site operator shape, parity split, or arrows: an explicit
  argument or dimension error before decomposition or contraction.

## Deferred Validation

CTMRG convergence is intentionally outside Tasks 4 and 5. It will be tested
later through the nested spinless-fermion example and the final acceptance
run. That later stage uses `D=2`, `chi=4,8,12`, `ctmrg_iter=20`, and
`ad_iter=20`, and checks that results trend toward the exact solution as
`chi` increases.
