# GrassmannTensorNetworks

GrassmannTensorNetworks provides Grassmann tensor data structures and operations in Julia, including contraction, fusion, decomposition, and ChainRules/Zygote automatic differentiation support through package extensions.

## Installation

### Use from the local checkout

If you are developing inside this repository, you can use the package directly with:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using GrassmannTensorNetworks
```

### Install into your default Julia environment

If you want to call it from a normal Julia session with `using GrassmannTensorNetworks`, add the repository once:

```julia
using Pkg
Pkg.add(url="https://github.com/Harrykjg-physics/Grassmanntn.jl.git")
using GrassmannTensorNetworks
```

If you want a live editable checkout instead of a fixed installed copy, use:

```julia
using Pkg
Pkg.develop(path="/path/to/GrassmannTensorNetworks")
using GrassmannTensorNetworks
```

## Registration status

This repository is already structured as a standard Julia package with `Project.toml` and `src/GrassmannTensorNetworks.jl`.
Registering it into the public Julia General registry still requires the JuliaRegistrator workflow on GitHub.

## Documentation

The repository includes a repository-hosted documentation site in [docs/](docs/index.md).
It covers the core Grassmann type, tensor operations, decomposition routines, and automatic differentiation support.
After enabling Read the Docs for this repository, the online version can be served from the project's Read the Docs URL.

## Development

From the package root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

The ChainRules/Zygote rules are loaded from `ext/GrassmannChainRulesCoreExt` when `ChainRulesCore` and `Zygote` are available. CUDA support is loaded from `ext/GrassmannCUDAExt` when `CUDA` is available.
