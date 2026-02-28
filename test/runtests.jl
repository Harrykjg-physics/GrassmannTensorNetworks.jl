# Add testets
using Test, TestExtras
using TensorOperations
using Shuffle
using LinearAlgebra
using TupleTools
using TupleTools: flatten, permute, insertat, insertafter, deleteat, getindices
using Random

Random.seed!(1234567)

Ti = time()

include("grassmann.jl")
include("fermionsign.jl")
include("base.jl")

Tf = time()

printstyled("Finished all tests in ", string(round((Tf - Ti) / 60; sigdigits=3)),
        " minutes."; bold=true, color=Base.info_color())
        