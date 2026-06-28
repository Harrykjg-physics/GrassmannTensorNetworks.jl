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