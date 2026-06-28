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

## Dependency order

The package loads these layers in the following order:

1. Core tensor types and tensor algebra from `src/`.
2. Utility and ansatz helpers from `auxiliary/`.
3. Model builders from `auxiliary/models.jl`.
4. CTMRG environment, iteration, and measurement routines.
5. Simple-update PEPS evolution.

This order matters because the algorithms depend on both the core `Grassmann` operations and the auxiliary PEPS/model helpers.