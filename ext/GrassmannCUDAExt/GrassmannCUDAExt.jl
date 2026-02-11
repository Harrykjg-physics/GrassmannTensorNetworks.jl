module GrassmannCUDAExt

using GrassmannTN
using CUDA

# Include CUDA-specific implementations
include("cuda_grassmann.jl")

end  # module GrassmannCUDAExt
