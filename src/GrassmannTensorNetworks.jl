module GrassmannTensorNetworks

using HDF5: create_group, h5open
using LinearAlgebra
using Printf
using Random
using Shuffle
using TensorOperations
using TupleTools
using TupleTools: flatten, permute, insertat, insertafter, deleteat, getindices
using VectorInterface

include("grassmann.jl")
include("fermionsign.jl")
include("base.jl")
include("linalg.jl")
include("contract.jl")
include("fusion.jl")
include("decomp.jl")
include("tupletools.jl")
include("auxiliary.jl")
include("algorithms.jl")

export Grassmann, AbstractGrassmann, GrassmannScalar, GrassmannVector, GrassmannMatrix
export _fixed_parity_blocks
export even, odd, data, index_type, tensor_parity, tensor_rank, scalar
export index_conjugation, nonzero_pairs, nonzero_keys, nonzero_vals

export auto_sign, trivial_sign, add_parity_sign, add_perm_sign

export trace, contract
export fuse
export gsvd, gevd, gortho
export save, load

export Nmod, compare_weights, prepare_bond_weight
export Square_GPEPS
export HubbardModel, nn_bond, gate
export CTMRGEnv, run_GCTMRG!, find_maxiter, read_CTMRG_env
export compute_exp_site, compute_exp_hbond, compute_exp_vbond
export correlation_function_horizontal, correlation_function_vertical
export Grassmann_SU

end