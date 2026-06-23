---
title: Decompositions and AD
---

# Decompositions and AD

## `gsvd`

### Matrix form

```julia
gsvd(tensor::GrassmannMatrix, Dcut; trunc=true, average_trunc=false)
```

Returns:

```julia
U, S, V, trunc_err
```

`gsvd` performs singular value decomposition sector by sector on the `(0, 0)` and `(1, 1)` blocks of a Grassmann matrix. It assumes row and column parity structure are compatible.

### Higher-rank form

```julia
gsvd(tensor, rowinds, colinds, Dcut; sign_function=trivial_sign, trunc=true, average_trunc=false)
```

This version:

1. Fuses `rowinds` into one row index.
2. Fuses `colinds` into one column index.
3. Applies matrix `gsvd`.
4. Splits the result back to tensor form.

Use this when a tensor should be decomposed as a map from one group of indices to another.

## `gevd`

### Matrix form

```julia
gevd(tensor::GrassmannMatrix, Dcut; symflag=false, trunc=true, average_trunc=false)
```

Returns:

```julia
U, Λ, trunc_err
```

`gevd` diagonalizes the parity blocks of a square Grassmann matrix. If `symflag=true`, each block is symmetrized before eigendecomposition.

### Higher-rank form

```julia
gevd(tensor, rowinds, colinds, Dcut; symflag=false, trunc=true, average_trunc=false)
```

This is the tensor analogue of matrix `gevd`, implemented by fuse-then-split around the matrix routine.

## `gortho`

### Matrix form

```julia
gortho(tensor::GrassmannMatrix; alg=LinearAlgebra.qr)
```

Returns:

```julia
M1, M2
```

Supported algorithms:

- `alg=LinearAlgebra.qr`: Grassmann QR-like factorization.
- `alg=LinearAlgebra.lq`: Grassmann LQ-like factorization.

### Higher-rank form

```julia
gortho(tensor, rowinds, colinds; alg=LinearAlgebra.qr)
```

Like the higher-rank `gsvd` and `gevd`, this groups row and column indices, performs the matrix factorization, and restores tensor structure afterward.

## `truncation`

Two truncation helpers exist:

```julia
truncation(S::GrassmannMatrix{Float64}, Dcut)
truncation(S::Vector{Float64}, size_even_k, Dcut)
```

They choose how many even and odd singular or eigen values survive a target cutoff while preserving parity bookkeeping.

## Automatic differentiation

The package ships AD rules through the extension:

```text
ext/GrassmannChainRulesCoreExt
```

When `ChainRulesCore` and `Zygote` are available, rules are loaded for core tensor operations and the decomposition routines.

This means workflows such as the following are intended to work:

```julia
using Grassmanntn
using Zygote

loss(A) = begin
    U, S, V, _ = gsvd(A, 1000; trunc=false)
    sum(U)
end

grad = gradient(loss, A)
```

## Practical guidance

- Use `trunc=false` when you want cleaner AD behavior and easier validation.
- Use `sign_function=trivial_sign` first when debugging tensor shapes.
- Switch to `auto_sign` when the fermionic ordering is part of the actual problem.
- For higher-rank decompositions, think of `rowinds` and `colinds` as the matrix row and column partition of the tensor.
