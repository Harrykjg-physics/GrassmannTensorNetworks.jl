# CPU/GPU conversion functions for Grassmann tensors

"""
    to_device(t::Grassmann, ::Val{:gpu}) -> Grassmann

Convert a Grassmann tensor from CPU (Array) to GPU (CuArray).
"""

function to_device(
    t::Grassmann{T, N}, 
    ::Val{:gpu}) where {T, N}

    # Convert each block from Array to CuArray
    cuda_data = Dict{NTuple{N, Int}, CuArray{T, N}}()
    sizehint!(cuda_data, length(data(t)))
    
    for (key, block) in t.data
        cuda_data[key] = CuArray(block)
    end
    
    return Grassmann(t.total_size, t.even_parity_size, t.index_type, cuda_data)
end

"""
    to_device(t::Grassmann, ::Val{:cpu}) -> Grassmann

Convert a Grassmann tensor from GPU (CuArray) to CPU (Array).
"""
function to_device(t::Grassmann{T, N}, ::Val{:cpu}) where {T, N}

    # Convert each block from CuArray to Array
    cpu_data = Dict{NTuple{N, Int}, Array{T, N}}()
    sizehint!(cpu_data, length(t.data))
    
    for (key, block) in t.data
        # Handle both CuArray and Array (already on CPU)
        if block isa CuArray
            cpu_data[key] = Array(block)
        else
            cpu_data[key] = block
        end
    end
    
    return Grassmann(t.total_size, t.even_parity_size, t.index_type, cpu_data)
end

"""
    cu(t::Grassmann) -> Grassmann

Convenience function to move Grassmann tensor to GPU.
Equivalent to `to_device(t, Val(:gpu))`.
"""
CUDA.cu(t::Grassmann) = to_device(t, Val(:gpu))
