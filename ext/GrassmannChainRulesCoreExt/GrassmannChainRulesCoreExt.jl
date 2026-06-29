module GrassmannChainRulesCoreExt

using ChainRulesCore
using LinearAlgebra
using Zygote
using GrassmannTensorNetworks

import GrassmannTensorNetworks: Grassmann, AbstractGrassmann, GrassmannScalar, GrassmannVector, GrassmannMatrix,
    nonzero_pairs, nonzero_keys, nonzero_vals, data,
    even, odd, index_type, tensor_parity, tensor_rank,
    trivial_sign, auto_sign,
    add_parity_sign, add_perm_sign,
    index_conjugation, prepare_range_dict,
    _parity_mask, _fixed_parity_blocks, _similar_arraytype,
    conjugate, fuse, calculate_sectors, calculate_fused_size, prepare_fused_info,
    trace, contract, gsvd, gevd, gortho, truncation, check_parity

# AD rules for grassmann.jl (constructors, convert, index_conjugation)
include("grassmann.jl")

# AD rules for fermionsign.jl (auto_sign, trivial_sign, add_parity_sign, add_perm_sign)
include("fermionsign.jl")

# AD rules for base.jl (copy, +, -, *, /, real, conj, permutedims, sqrt, convert(Array, ...))
include("base.jl")

# AD rules for linalg.jl (log, norm, diag, transpose, inv, dot)
include("linalg.jl")

# AD rules for contract.jl (trace and contract)
include("contract.jl")

# AD rules for fusion.jl (fuse, split)
include("fusion.jl")

# AD rules for decomp.jl (gsvd, gevd, gortho)
include("decomp.jl")

end
