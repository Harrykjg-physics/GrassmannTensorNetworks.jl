# Grassmanntn

Grassmanntn provides Grassmann tensor data structures and operations in Julia, including contraction, fusion, decomposition, and ChainRules/Zygote automatic differentiation support through package extensions.

## Development

From the package root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

The ChainRules/Zygote rules are loaded from `ext/GrassmannChainRulesCoreExt` when `ChainRulesCore` and `Zygote` are available. CUDA support is loaded from `ext/GrassmannCUDAExt` when `CUDA` is available.
