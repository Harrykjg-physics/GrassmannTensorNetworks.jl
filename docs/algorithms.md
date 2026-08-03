# Algorithms and Helpers

The package now ships two higher-level layers on top of the core tensor algebra.

## Auxiliary types

`Square_GPEPS` stores a square-lattice Grassmann PEPS unit cell together with optional bond weights:

```julia
Square_GPEPS{T}
```

Its main fields are:

- `A`: rank-5 local tensors.
- `Λx`, `Λy`: optional nearest-neighbor bond weights.

The package also provides matching HDF5 methods:

```julia
save(peps, filename, group)
load(filename, group, Square_GPEPS; has_bond_weights=true)
```

## Model helpers

`HubbardModel` builds nearest-neighbor objects for square-lattice Fermi-Hubbard workflows:

```julia
model = HubbardModel(t, U, μ)
Hbond = nn_bond(model)
G = gate(model, dτ)
```

Use `nn_bond` for the two-site operator and `gate` for the imaginary-time evolution gate.

## Simple update

The nearest-neighbor simple-update routine is:

```julia
Grassmann_SU(G, peps, dτ, Dbond; su_iter=1000, su_tol=1e-12)
```

It updates a `Square_GPEPS` in place using repeated x-bond and y-bond sweeps.

## CTMRG

`CTMRGEnv` stores the edge and corner tensors used by corner transfer matrix renormalization:

```julia
CTMRGEnv(...)
```

The main entry point is:

```julia
run_GCTMRG!(T_bulk, T_imp, env, χ; ctmrg_iter=100, ctmrg_tol=1e-12)
```

Useful helpers include:

- `find_maxiter`
- `read_CTMRG_env`
- `compute_exp_site`
- `compute_exp_hbond`
- `compute_exp_vbond`
- `correlation_function_horizontal`
- `correlation_function_vertical`

## Nested CTMRG

### Network layout and CTMRG reuse

`NestedLayout` records the source and doubled unit-cell sizes and maps every
source site to its K, Y, X, and B positions. `NestedNetwork` stores the
resulting rank-4 tensor matrix, its layout, and the final bulk X tensors. The
public constructors and main builder are:

```julia
layout = NestedLayout((Lx, Ly))
layout = NestedLayout(peps)
nested = NestedNetwork(network, layout, x_crossings)
nested = nested_network(peps)
nested = nested_network(peps, layout)
```

For an `Lx x Ly` `Square_GPEPS`, this constructs a `2Lx x 2Ly` checkerboard.
The first source coordinate grows downward and the second grows rightward.
Each physical source tensor is represented by the periodic source-centred cell
`[B X; Y K]`: X is to the right of B, and Y is below B. The stored
Grassmann arrows follow the CTMRG input convention: horizontally right-to-left
and vertically top-to-bottom.
`NestedNetwork` supports `size`, `axes`, and indexing like its
underlying tensor matrix. The three-argument `NestedNetwork` constructor wraps
an already assembled compatible network; `nested_network` performs the K/Y/X/B
construction and link checks.

The nested network uses the existing CTMRG environment and iteration code:

```julia
env = initialize_nested_environment(nested, chi)
run_nested_GCTMRG!(nested, env, chi; ctmrg_iter=20, ctmrg_tol=1e-10)
```

`initialize_nested_environment` also accepts an optional `chi_even`, which
defaults to `div(chi, 2)`. `run_nested_GCTMRG!` forwards keyword arguments to
`run_GCTMRG!`, checks that the network and environment unit cells match, and
updates and returns `env`.

### Measurements

`nested_x_operator(nested, peps, site, operator)` constructs the X tensor with
a rank-2 physical operator inserted at `site`. The operator must match the
local physical dimension and parity split and have arrows `(:out, :in)`.
`nested_y_operator` remains as a compatibility alias for the former public API.

The public expectation-value routines are:

```julia
compute_nested_exp_site(nested, peps, operator, env, site)
compute_nested_exp_hbond(nested, peps, operator, env, site)
compute_nested_exp_vbond(nested, peps, operator, env, site)

compute_nested_exp_site(nested, peps, operators, env)
compute_nested_exp_hbond(nested, peps, operators, env)
compute_nested_exp_vbond(nested, peps, operators, env)
```

Each returns `(denominator, value)`, where `value` is the normalized
expectation value. The site operator is rank 2. Bond operators are rank 4,
have arrows `(:out, :out, :in, :in)`, and must have even total parity. In the
no-`site` overloads, `operators` is a matrix matching the PEPS unit cell; the
result is `(denominators, values)`, two matrices with that same unit-cell size.
Horizontal and vertical bond routines use the periodic right and lower
neighbor, respectively.

The exact one-site and nearest-neighbor tests do not initialize or iterate a
CTMRG environment. They reblock each `[B X; Y K]` cell, compare every tensor
element, block metadata, maximum element error, absolute norm error, and
relative norm error with `reduced_tensor`, and then close the reblocked and
reduced networks for all four boundary twists. Measurement tests compare the
denominator, numerator, and normalized ratio.

### Fermionic signs and representation

The nested construction keeps all signs in the existing Grassmann operations:

- K and B use explicit parity signs on the legs crossed by the chosen planar
  routing. Their fused legs are ordered as `(physical, virtual)`.
- X carries the physical identity or impurity. Its internal crossing uses
  `add_parity_sign` and `add_perm_sign`. An operator of total parity `q`
  additionally contributes `(-1)^(q*(1+u))` before fusion; this is unity for
  bulk/even operators and is required for odd Schmidt endpoints. Y is a pure
  virtual crossing built from identities, `index_conjugation`, and
  `add_parity_sign`.
- A horizontal operator-Schmidt term has no additional endpoint-exchange
  factor. A vertical odd term receives
  `(-1)^tensor_parity(top_operator)`.
- Conjugation, index bending, contraction, fusion, and permutation continue
  to use the package `global_sign` convention.

No MPO representation or new Grassmann tensor primitive is introduced.
`NestedNetwork` is a lightweight layout wrapper around the existing rank-4
`Grassmann` tensors, and the implementation uses `add_parity_sign`,
`add_perm_sign`, conjugation, contraction, and fusion.

### Automatic differentiation

Nested reverse rules are defined in
`algorithms/Nested_CTMRG/nested_chainrules.jl` and loaded only through
`GrassmannChainRulesCoreExt`. Loading `ChainRulesCore` and `Zygote` activates
them without adding either package to the core dependency path. The extension
provides rules for `nested_network` and operator-dressed `nested_x_operator`
construction.

The public horizontal and vertical measurement rules use a fixed-observable
contract: they preserve cotangents for `nested` and `env`, while `peps`, the
bond operator, and the site are treated as static. Consequently these public
bond rules do not support gradients with respect to the operator. For a site
measurement with a fixed boundary and fixed nested network, the direct PEPS
argument has a structural-zero derivative; the rank-2 operator derivative is
supported and tested. Rebuild `nested_network(peps)` inside the differentiated
objective when PEPS-tensor gradients are required.

## Dependency order

The package loads these layers in the following order:

1. Core tensor types and tensor algebra from `src/`.
2. Utility and ansatz helpers from `auxiliary/`.
3. Model builders from `auxiliary/models.jl`.
4. CTMRG environment, iteration, and measurement routines.
5. Nested-network construction and measurement routines.
6. Simple-update PEPS evolution.

This order matters because the algorithms depend on both the core `Grassmann` operations and the auxiliary PEPS/model helpers.
