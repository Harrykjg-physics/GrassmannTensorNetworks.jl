# GrassmannTensorNetworks

GrassmannTensorNetworks is a Julia package for working with Grassmann tensors with Z2 parity structure.
It provides:

- A block-structured `Grassmann` tensor type.
- Fermionic sign handling through `trivial_sign` and `auto_sign`.
- Tensor trace and contraction operations.
- Index fusion and splitting.
- Matrix and higher-rank decompositions: `gsvd`, `gevd`, and `gortho`.
- Automatic differentiation support through the `ChainRulesCore` and `Zygote` extension.

## What the package models

Each tensor stores only the parity-allowed blocks. A `Grassmann` tensor keeps:

- `total_size`: the full size of each index.
- `even_parity_size`: how many states of each index are even.
- `index_type`: whether an index is ordinary (`:in`) or dual (`:out`).
- `data`: the nonzero parity blocks keyed by their Z2 sector.

This layout is useful when fermionic parity is part of the tensor algebra and should be preserved by construction.

## Documentation map

- [Core Type and Helpers](grassmann.md)
- [Tensor Operations](operations.md)
- [Decompositions and AD](decompositions.md)

## Quick start

```julia
using GrassmannTensorNetworks

T = Grassmann(
    (4, 4, 4),
    (2, 2, 2),
    (:out, :in, :in),
    Float64;
    init=:random,
    parity=:even,
)

U, S, V, err = gsvd(fuse(T, (1, 2)), 8; trunc=false)
```

For development and tests:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

## Publishing

This site lives in the repository `docs/` directory and can be published with either GitHub Pages or Read the Docs.
