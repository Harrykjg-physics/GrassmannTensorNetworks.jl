---
title: Core Type and Helpers
---

# Core Type and Helpers

## `Grassmann`

The central type is:

```julia
Grassmann{T, N, AT<:AbstractArray{T, N}}
```

It represents an `N`-rank tensor with explicit Z2 parity structure. Only parity-allowed blocks are stored.

### Main constructors

```julia
Grassmann(total_size, even_parity_size, index_type, data)
Grassmann(total_size, even_parity_size, index_type, T; init=:random, parity=:even)
Grassmann(array, total_size, even_parity_size, index_type; parity=:even)
```

Typical use:

```julia
T = Grassmann((4, 4), (2, 2), (:out, :in), Float64; init=:random, parity=:even)
```

Use the array-based constructor when you already have a dense tensor and want to project it into the stored parity blocks.

## Exported aliases

- `GrassmannScalar`
- `GrassmannVector`
- `GrassmannMatrix`

These are convenience aliases for rank-0, rank-1, and rank-2 tensors.

## Structure queries

The package exports a compact set of helpers for inspecting tensor structure:

- `even(t)`: even-parity size tuple.
- `odd(t)`: odd-parity size tuple.
- `data(t)`: underlying block dictionary.
- `index_type(t)`: tuple of `:in` and `:out`.
- `tensor_parity(t)`: total parity of the tensor.
- `tensor_rank(t)`: rank of the tensor.
- `scalar(t)`: convert a rank-0 tensor to a scalar value.
- `nonzero_pairs(t)`, `nonzero_keys(t)`, `nonzero_vals(t)`: iterate over stored blocks.

These are the right entry points if you want to build tooling around the package rather than accessing fields manually.

## Index conjugation

```julia
index_conjugation(t, inds)
index_conjugation(t, ind)
```

This toggles selected index types between `:in` and `:out`. It is useful when preparing tensors for contractions that depend on Grassmann dual structure.

## Sign helpers

Grassmanntn separates tensor algebra from sign bookkeeping:

- `trivial_sign(args...)`: always returns `1`.
- `auto_sign(...)`: computes fermionic signs from sectors, contraction pattern, boundary condition, or permutation.
- `add_parity_sign(t, ind; sign_function=auto_sign)`: multiplies by a parity sign on one index.
- `add_perm_sign(t, dst; sign_function=auto_sign)`: multiplies by the sign induced by a permutation.

Use `trivial_sign` for purely structural testing and `auto_sign` when physical fermionic ordering matters.

## Save and load

```julia
save(filename, group, t)
load(filename, group, Grassmann)
```

These helpers serialize a tensor to HDF5. The current implementation stores:

- `total_size`
- `even_size`
- `index_type`
- dense tensor data
- tensor parity

This is convenient for experiments, checkpoints, and exchanging tensors with external workflows.
