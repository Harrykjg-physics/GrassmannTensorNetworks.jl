module GrassmannChainRulesCoreExt

using ChainRulesCore
using LinearAlgebra
using Zygote
using GrassmannTensorNetworks

import GrassmannTensorNetworks: Grassmann, AbstractGrassmann, GrassmannScalar, GrassmannVector, GrassmannMatrix,
    Square_GPEPS,
    nonzero_pairs, nonzero_keys, nonzero_vals, data,
    even, odd, index_type, tensor_parity, tensor_rank,
    trivial_sign, auto_sign,
    add_parity_sign, add_perm_sign,
    index_conjugation, prepare_range_dict,
    _parity_mask, _fixed_parity_blocks, _similar_arraytype,
    conjugate, fuse, calculate_sectors, calculate_fused_size, prepare_fused_info,
    trace, contract, gsvd, gevd, gortho, truncation, check_parity

import GrassmannTensorNetworks:
    NestedLayout, NestedNetwork,
    nested_network, nested_y_operator,
    compute_nested_exp_hbond, compute_nested_exp_vbond,
    _source_site, _graded_pair_sign,
    _nested_ket, _nested_ket_raw,
    _nested_bra, _nested_bra_raw, _bend_index,
    _nested_x,
    _nested_ket_for_network, _nested_bra_for_network,
    _nested_x_for_network, _nested_y_for_network,
    _nested_y_operator_raw,
    _physical_identity, _operator_schmidt,
    _nested_scalar_or_zero,
    _contract_nested_hpatch3, _contract_nested_vpatch3

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

include(joinpath(
    @__DIR__, "..", "..", "algorithms", "Nested_CTMRG", "nested_chainrules.jl"
))

end
