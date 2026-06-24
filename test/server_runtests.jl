using Pkg

const PKG_ROOT = normpath(joinpath(@__DIR__, ".."))

Pkg.activate(mktempdir())
Pkg.develop(path=PKG_ROOT)
Pkg.add(PackageSpec(name="TestExtras", version="0.1"))
Pkg.add(PackageSpec(name="ChainRulesCore", version="1"))
Pkg.add(PackageSpec(name="FiniteDifferences", version="0.12"))
Pkg.add(PackageSpec(name="Shuffle", version="0.1"))
Pkg.add(PackageSpec(name="TensorOperations"))
Pkg.add(PackageSpec(name="TupleTools"))
Pkg.add(PackageSpec(name="Zygote", version="0.7"))

using Test
using TestExtras
using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using Random
using Shuffle: shuffle
using TensorOperations
using TupleTools
using TupleTools: flatten, permute, insertat, insertafter, deleteat, getindices
using Zygote
using Zygote: bufferfrom
using GrassmannTensorNetworks

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
printstyled(
    "Finished all tests in ",
    string(elapsed_minutes),
    " minutes.";
    bold = true,
    color = Base.info_color(),
)
