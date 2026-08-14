# GrassmannTensorNetworks

GrassmannTensorNetworks is a Julia package for Grassmann tensor networks with Z2 parity structure.
It includes core tensor algebra, decomposition routines with AD support, and higher-level PEPS / CTMRG utilities.

## Installation

```julia
using Pkg
Pkg.add("GrassmannTensorNetworks")
using GrassmannTensorNetworks
```

For local development:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using GrassmannTensorNetworks
```

## Package layout

The package currently has three layers:

- `src/`: core Grassmann tensor types and tensor algebra.
- `auxiliary/`: PEPS ansatz, model helpers, and utility functions.
- `algorithms/`: simple update and CTMRG routines built on top of the core package API.

## Documentation

The repository docs live in [docs/](docs/index.md).
They cover:

- the `Grassmann` type and structural helpers,
- tensor operations and decompositions,
- AD support,
- higher-level PEPS / CTMRG utilities.

## Development

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

The ChainRules/Zygote rules are loaded from `ext/GrassmannChainRulesCoreExt` when `ChainRulesCore` and `Zygote` are available.
CUDA support is loaded from `ext/GrassmannCUDAExt` when `CUDA` is available.

## Examples

- `examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl`:
  Grassmann nested-CTMRG optimization for the square-lattice spinless fermion.

Run the nested example from the repository root with, for example:

```powershell
$env:NESTED_CHI = "4"
$env:NESTED_SEED = "1234"
$env:NESTED_VERBOSITY = "0"
julia --project=. examples/Spinless_Fermion_2D_Square_AD_nested/Spinless_Fermion_2D_Square_AD_nested.jl
```

The executable configuration fixes `D = 2`, `ctmrg_iter = 20`, and
`ad_iter = 20`. The requested acceptance runs set `NESTED_CHI` to `4`, `8`,
and `12` with a common seed. Their finite, error-free energies should show an
overall trend toward the exact value `-6.170521774015...` as `chi` increases;
there is no fixed one-percent error threshold at `chi = 12`.
