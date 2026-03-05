module Grassmanntn

using TensorOperations
using TupleTools
using TupleTools:flatten, permute, insertat, insertafter, deleteat, getindices
using VectorInterface
using Shuffle
using ChainRulesCore, Zygote
using Zygote:bufferfrom
using LinearAlgebra

include("grassmann.jl")
include("fermionsign.jl")
include("base.jl")
include("contract.jl")
include("fusion.jl")

include("../ext/tupletools.jl")

include("../ext/GrassmannChainRulesCoreExt/grassmann.jl")
include("../ext/GrassmannChainRulesCoreExt/fermionsign.jl")
include("../ext/GrassmannChainRulesCoreExt/base.jl")
include("../ext/GrassmannChainRulesCoreExt/contract.jl")
include("../ext/GrassmannChainRulesCoreExt/fusion.jl")

export Grassmann, _fixed_parity_blocks
export even, data, index_type, tensor_parity, tensor_rank, scalar, index_conjugation, nonzero_pairs, nonzero_keys, nonzero_vals

export auto_sign, trivial_sign, add_parity_sign, add_perm_sign

export trace, contract
export fuse

end


