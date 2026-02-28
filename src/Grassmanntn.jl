module Grassmanntn

using TensorOperations
using TupleTools
using TupleTools: flatten, permute, insertat, insertafter, deleteat, getindices
using VectorInterface
using ChainRulesCore, Zygote
using LinearAlgebra

include("grassmann.jl")
include("fermionsign.jl")
include("base.jl")
include("contract.jl")

include("../ext/GrassmannChainRulesCoreExt/grassmann.jl")
include("../ext/GrassmannChainRulesCoreExt/fermionsign.jl")
include("../ext/GrassmannChainRulesCoreExt/base.jl")
include("../ext/GrassmannChainRulesCoreExt/contract.jl")

end


