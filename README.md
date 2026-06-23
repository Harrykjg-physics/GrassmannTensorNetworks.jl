# Grassmanntn

Grassmanntn provides Grassmann tensor data structures and operations in Julia, including contraction, fusion, decomposition, and ChainRules/Zygote automatic differentiation support through package extensions.

## Documentation

The repository includes a repository-hosted documentation site in [docs/](docs/index.md).
It covers the core Grassmann type, tensor operations, decomposition routines, and automatic differentiation support.
After enabling Read the Docs for this repository, the online version can be served at [Grassmanntn.readthedocs.io/en/latest/](https://Grassmanntn.readthedocs.io/en/latest/).

## Development

From the package root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

The ChainRules/Zygote rules are loaded from `ext/GrassmannChainRulesCoreExt` when `ChainRulesCore` and `Zygote` are available. CUDA support is loaded from `ext/GrassmannCUDAExt` when `CUDA` is available.

