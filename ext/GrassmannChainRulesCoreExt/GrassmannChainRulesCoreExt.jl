module GrassmannChainRulesCoreExt

using GrassmannTN
using ChainRulesCore
using Zygote

import GrassmannTN: Grassmann, AbstractGrassmann,
    nonzero_pairs, nonzero_keys, nonzero_vals, data,
    even, odd, index_type, tensor_parity, tensor_rank,
    trivial_sign, auto_sign,
    add_parity_sign, add_perm_sign,
    index_conjugation, convert2array, prepare_range_dict,
    _parity_mask, _fixed_parity_blocks, _similar_arraytype,
    conjugate, fuse, calculate_sectors, calculate_fused_size, prepare_fused_info

# AD rules for grassmann.jl (constructors, convert, index_conjugation)
include("grassmann.jl")

# AD rules for fermionsign.jl (auto_sign, trivial_sign, add_parity_sign, add_perm_sign)
include("fermionsign.jl")

# AD rules for base.jl (copy, +, -, *, /, real, conj, permutedims, sqrt, convert2array)
include("base.jl")

# AD rules for fusion.jl (fuse, split)
include("fusion.jl")

end  # module GrassmannChainRulesCoreExt
