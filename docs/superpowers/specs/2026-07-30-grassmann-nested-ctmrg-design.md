# Grassmann Nested CTMRG Design

## Objective

Add a native Grassmann implementation of the doubled-cell nested tensor
network to GrassmannTensorNetworks. The implementation must:

- map every tensor of a square-lattice `Square_GPEPS` unit cell to a `2 x 2`
  checkerboard of ket (`K`), physical-route (`Y`), virtual-crossing (`X`),
  and bra (`B`) tensors;
- contract the resulting `2Lx x 2Ly` four-leg network with the existing
  `CTMRGEnv` and `run_GCTMRG!` implementation;
- evaluate normalized one-site and horizontal or vertical nearest-neighbor
  two-site operators;
- support reverse-mode automatic differentiation back to the source PEPS;
- provide a spinless-fermion AD example and the requested numerical
  acceptance runs; and
- preserve the current package structure and dependencies as far as possible.

The mathematical and sign conventions follow the supplied handwritten
diagrams, the monograph `0730_Nested fTN/main.pdf`, and the local
`PEPSKit_nesting` reference implementation. The new implementation uses this
repository's `Grassmann` tensors and does not introduce PEPSKit or TensorKit as
dependencies.

## Source Organization

The implementation is localized under:

```text
algorithms/
  Nested_CTMRG/
    nested_network.jl
    measurements.jl
    nested_chainrules.jl
examples/
  Spinless_Fermion_2D_Square_AD_nested/
    Project.toml
    Spinless_Fermion_2D_Square_AD_nested.jl
```

`src/algorithms.jl` includes `nested_network.jl` and `measurements.jl`.
`nested_chainrules.jl` is included from
`ext/GrassmannChainRulesCoreExt/GrassmannChainRulesCoreExt.jl`, so
ChainRulesCore and Zygote remain optional dependencies.

Tests are added to the existing `test/` tree and included from
`test/runtests.jl`. No existing CTMRG move or projector implementation is
copied.

## Network Representation

`NestedLayout` records the source size, doubled size, and four source-to-nested
coordinate maps. For source site `(r, c)`:

| Nested coordinate | Node | Responsibility |
| --- | --- | --- |
| `(2r - 1, 2c - 1)` | `K` | Ket tensor with the physical leg routed east |
| `(2r - 1, 2c)` | `Y` | Physical identity/operator route plus graded crossing |
| `(2r, 2c - 1)` | `X` | Pure graded virtual crossing |
| `(2r, 2c)` | `B` | Conjugate bra tensor with the physical leg routed north |

`NestedNetwork` stores:

- a `Matrix{Grassmann{T,4}}` accepted directly by the existing CTMRG code;
- its `NestedLayout`; and
- any static crossing metadata required by measurement and reverse rules.

It delegates `size`, `axes`, and indexing to the nested tensor matrix. A
supplied layout must match the source PEPS unit-cell size.

The four nodes use the same external order as existing CTMRG bulk tensors:
left, right, up, down. Neighboring legs must have equal total/even dimensions
and dual-compatible `:in`/`:out` orientations, including across periodic
boundaries.

`initialize_nested_environment` constructs
`CTMRGEnv(nested.network, chi, chi_even)`. Boundary contraction calls
`run_GCTMRG!(nested.network, nested.network, env, chi; ...)`; the nested
implementation does not maintain a second CTMRG engine.

## Fermionic Sign Invariants

Fermionic signs are part of the public correctness contract.

- Tensor axes are never reordered as ordinary dense-array bookkeeping.
  Reordering uses Grassmann `permutedims` with
  `sign_function=global_sign`.
- Fusion, conjugation, and arrow reversal use `fuse`, Grassmann `conj`,
  `add_perm_sign`, `add_parity_sign`, and `index_conjugation` as required by
  the diagrammatic ordering.
- `X` implements the graded exchange
  `(-1)^(p_horizontal * p_vertical)` block by block. Its odd-odd block is
  exactly `-1`; the other three parity combinations are `+1`.
- `Y` combines the same graded exchange with the physical identity. An
  observable-dressed `Y` replaces only that physical identity with the
  supplied operator.
- K/B physical-leg fusion includes the explicit exchange sign required before
  fusion. X/Y crossings and orientation twists account for the remaining
  line intersections.
- Periodic neighbor lookup uses the same local orientation and sign
  convention as the interior cell.

These rules are checked algebraically rather than accepted only from visual
agreement with the reference diagrams.

## Observables

The public measurement interface is:

```julia
compute_nested_exp_site(...)
compute_nested_exp_hbond(...)
compute_nested_exp_vbond(...)
```

The functions return the closed-network denominator and the normalized
expectation value, matching the style of the existing `compute_exp_*`
functions. Both individual-site inputs and source-unit-cell matrix inputs are
supported where the current CTMRG APIs support them.

### One-site operators

The source site is mapped to its `Y` coordinate. The physical identity inside
that Y tensor is replaced by the one-site operator. The numerator is the
corresponding one-site CTMRG contraction; the denominator is the identical
contraction with the closed Y tensor.

### Two-site operators

Nearest-neighbor source sites are separated by one node in the doubled cell.
The minimal observable patches are:

- horizontal: `Y - K - Y`;
- vertical: `Y - B - Y`.

The two open physical routes are contracted with the supplied two-site
operator while the intermediate K or B tensor remains part of the patch. The
denominator uses the same patch with the two-site physical identity. This
keeps numerator and denominator contraction order and fermionic sign
conventions identical.

Only one-site and horizontal or vertical nearest-neighbor source geometries
are supported. Unit-cell sizes, mapped coordinates, physical dimensions,
parity splits, index orientations, operator ranks, and environment sizes are
validated before contraction. Unsupported geometries raise `ArgumentError`;
dimension and parity incompatibilities raise `DimensionMismatch`.

## Reverse-Mode Differentiation

The differentiable data path is:

```text
PEPS parameters
  -> Square_GPEPS tensors
  -> K and B nested tensors
  -> fixed CTMRG environment measurement
  -> energy
```

Layout coordinates, X crossings, Y crossing geometry, and identity/fusion
bases depend on spaces rather than the numerical PEPS entries and therefore
receive `NoTangent`.

The `rrule` for nested-network construction:

1. obtains local reverse rules for each differentiable K and B construction;
2. extracts nested tensor cotangents at the corresponding layout positions;
3. applies the K and B pullbacks;
4. materializes and adds both cotangents for each source tensor; and
5. returns a `Square_GPEPS`-compatible cotangent while assigning no tangent to
   static layout and crossing metadata.

Measurement construction receives narrow reverse rules only where Zygote
cannot trace the existing Grassmann primitives. Existing package rules for
`Grassmann`, `contract`, `trace`, `fuse`, `permutedims`, and conjugation are
reused rather than duplicated.

The example optimizes a fixed-environment objective with Zygote gradients.
After each outer AD step, CTMRG is refreshed and the candidate is validated
with a refreshed environment, following the trust-region/backtracking
structure of the existing spinless-fermion AD example.

## Spinless-Fermion Example

`Spinless_Fermion_2D_Square_AD_nested.jl` exposes a callable driver and a
script entry point. Defaults are:

```text
t = 1
gamma = 1
lambda = 3
Lx = Ly = 2
D = 2
ctmrg_iter = 20
ad_iter = 20
```

The environment dimension, seed, iteration counts, output path, and verbosity
are configurable through function arguments and environment variables so the
three acceptance runs can execute independently.

The exact thermodynamic-limit energy density is evaluated by Brillouin-zone
quadrature:

```math
e_0 = -\lambda -
\frac{1}{4\pi^2}\int_{-\pi}^{\pi}\int_{-\pi}^{\pi}
\sqrt{(t(\cos k_x+\cos k_y)-\lambda)^2
      +\gamma^2(\sin k_x+\sin k_y)^2}\,dk_x\,dk_y.
```

For the default parameters this is approximately
`-6.170521774015` per site. Each run records the final energy, exact energy,
signed/absolute/relative errors, CTMRG diagnostics, optimization history,
elapsed time, and allocation or memory information available to the driver.

## Verification

Verification proceeds from exact local algebra to the requested long runs:

1. `NestedLayout` coordinate maps, node sizes, parity splits, index arrows,
   and periodic neighbor compatibility.
2. A pure odd one-dimensional X crossing has scalar value `-1`.
3. Contracting a complete K/Y/X/B tile equals the existing
   `reduced_tensor(A)` after applying one documented external-leg
   identification and ordering. Tests cover real and complex tensors and
   nontrivial even/odd sectors.
4. Closing an open Y tensor with the physical identity reconstructs the
   original closed Y tensor.
5. One-site and two-site identity expectations equal one.
6. General one-site and nearest-neighbor values agree with the existing
   reduced-layer measurement on deterministic small instances.
7. Directional derivatives of nested-network construction and one-site and
   two-site energy paths agree with centered finite differences to relative
   error at most `1e-4`.
8. The existing package test suite remains green.
9. On the required Julia SSH server, run three independent tasks with
   `D=2`, `chi=4,8,12`, `ctmrg_iter=20`, and `ad_iter=20`, under the required
   20 GB memory guard.

Numerical acceptance requires all three long runs to finish without errors
and produce finite energies. The result table must show that increasing
`chi` gives an overall trend toward the exact energy. No fixed final relative
error threshold is imposed. Because finite-iteration stochastic optimization
can introduce small nonmonotonic fluctuations, the review considers the
three-run trend and diagnostics together rather than requiring every adjacent
energy difference to be strictly monotonic.

## Pull Request Gate

Only after local tests, gradient checks, and all three server acceptance runs
pass is the branch pushed and a GitHub pull request opened as Codex. The pull
request includes:

- a summary of the nested construction and sign conventions;
- the exact commands and environment variables used for acceptance;
- the `chi=4,8,12` energy/error table;
- CTMRG and AD iteration settings;
- test and gradient-check results; and
- any observed limitations.

## Explicit Non-Goals

- No PEPSKit or TensorKit dependency.
- No duplicate CTMRG implementation.
- No anyonic or non-symmetric braiding.
- No long-range or diagonal two-site observables in the first version.
- No unrelated refactoring of the existing CTMRG, simple-update, or core
  Grassmann algebra.
