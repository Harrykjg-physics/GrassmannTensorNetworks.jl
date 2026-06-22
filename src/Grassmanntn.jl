module Grassmanntn

using HDF5: create_group, h5open
using LinearAlgebra
using Random
using Shuffle
using TensorOperations
using TupleTools
using TupleTools: flatten, permute, insertat, insertafter, deleteat, getindices
using VectorInterface

include("grassmann.jl")
include("fermionsign.jl")
include("base.jl")
include("contract.jl")
include("fusion.jl")
include("decomp.jl")
include("tupletools.jl")

export Grassmann, AbstractGrassmann, GrassmannScalar, GrassmannVector, GrassmannMatrix
export _fixed_parity_blocks
export even, odd, data, index_type, tensor_parity, tensor_rank, scalar
export index_conjugation, nonzero_pairs, nonzero_keys, nonzero_vals

export auto_sign, trivial_sign, add_parity_sign, add_perm_sign

export trace, contract
export fuse
export gsvd, gevd, gortho

end
