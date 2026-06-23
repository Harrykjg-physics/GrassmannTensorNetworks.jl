using Test
using TestExtras
using GrassmannTensorNetworks
using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using Random
using Shuffle
using TensorOperations
using TupleTools
using TupleTools: flatten, permute, insertat, insertafter, deleteat, getindices
using Zygote
using Zygote: bufferfrom

import GrassmannTensorNetworks: truncation

Random.seed!(1234567)

start_time = time()

@testset "GrassmannTensorNetworks" begin
    include("grassmann.jl")
    include("fermionsign.jl")
    include("base.jl")
    include("fusion.jl")
    include("contract.jl")
    include("decomp.jl")
end

elapsed_minutes = round((time() - start_time) / 60; sigdigits=3)
printstyled("Finished all tests in ", string(elapsed_minutes), " minutes."; bold=true, color=Base.info_color())
