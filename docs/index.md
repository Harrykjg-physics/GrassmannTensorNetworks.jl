# GrassmannTensorNetworks

GrassmannTensorNetworks is a Julia package for Z2-graded Grassmann tensor algebra.

It has three layers:

- Core tensor data structures and parity-aware tensor operations.
- Decomposition routines with ChainRules/Zygote support.
- Higher-level PEPS, simple-update, model, and CTMRG utilities.

## Quick start

```julia
using GrassmannTensorNetworks

T = Grassmann((4, 4, 4), (2, 2, 2), (:out, :in, :in), Float64; init=:random, parity=:even)
U, S, V, err = gsvd(fuse(T, (1, 2)), 8; trunc=false)
```

## Documentation map

- [Core Type and Helpers](grassmann.md)
- [Tensor Operations](operations.md)
- [Decompositions and AD](decompositions.md)
- [Algorithms and Helpers](algorithms.md)

## Development

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```