# Grassmanntn

Grassmanntn provides Grassmann tensor data structures and operations in Julia, including contraction, fusion, decomposition, and ChainRules/Zygote automatic differentiation support through package extensions.

## Documentation

The repository includes a GitHub Pages friendly documentation site in [docs/](docs/index.md).
It covers the core Grassmann type, tensor operations, decomposition routines, and automatic differentiation support.

## Development

From the package root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

The ChainRules/Zygote rules are loaded from `ext/GrassmannChainRulesCoreExt` when `ChainRulesCore` and `Zygote` are available. CUDA support is loaded from `ext/GrassmannCUDAExt` when `CUDA` is available.

