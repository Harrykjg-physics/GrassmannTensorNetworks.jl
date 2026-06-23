---
title: Tensor Operations
---

# Tensor Operations

## `trace`

```julia
trace(T, inds_tr; sign_function=trivial_sign, cj=false, perm=..., pbc=...)
```

`trace` contracts pairs of indices within a single tensor.

There are two public calling styles:

```julia
trace(T, (i, j))
trace(T, ((i1, i2, ...), (j1, j2, ...)))
```

Important options:

- `sign_function`: use `auto_sign` when fermionic sign tracking matters.
- `cj`: whether the tensor is conjugated before tracing.
- `perm`: permutation applied to the output open indices.
- `pbc`: periodic or anti-periodic boundary-condition signs for traced indices.

Use `trace` when you want to eliminate internal degrees of freedom from a single Grassmann tensor.

## `contract`

```julia
contract(T1, T2; sign_function=trivial_sign, perm=..., cj=(false, false))
contract(T1, T2, (i, j); ...)
contract(T1, T2, ((i1, i2, ...), (j1, j2, ...)); ...)
```

`contract` is the main binary tensor contraction interface.

Supported modes:

- Direct product with no contracted indices.
- Single-index contraction.
- Multi-index contraction.

Important options:

- `sign_function`: choose between structural and fermionic-sign-aware contraction.
- `perm`: reorder the output open indices.
- `cj`: independently conjugate `T1` and `T2` before contraction.

If you are building tensor-network style code, `contract` is usually the main workhorse.

## `fuse`

```julia
fuse(tensor, inds; index_type_fused=:in)
```

`fuse` combines several nearby indices into one Grassmann index while preserving Z2 structure. The fused indices must be consecutive and listed in increasing order.

This is especially useful before applying matrix-style decompositions to higher-rank tensors.

Example:

```julia
Tf = fuse(T, (1, 2))
```

## `split`

```julia
split(tensor, ind, total_size_split, even_size_split, index_type_split)
```

`split` is the inverse companion to `fuse`. It reconstructs multiple indices from one fused index using the target tensor metadata.

Although `split` is defined as `Base.split`, in practice you will mostly encounter it indirectly inside higher-rank `gsvd`, `gevd`, and `gortho`.

## When to use `fuse` and `split`

Use them when:

- You want to turn a high-rank tensor into a matrix-like object.
- You need to apply SVD, EVD, QR, or LQ by grouping rows and columns.
- You want to preserve parity-aware block structure during reshaping.

Avoid manual dense reshapes when you want Grassmann sector bookkeeping to stay correct.
