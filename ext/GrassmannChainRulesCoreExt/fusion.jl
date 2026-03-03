################################ AD rules for fusion.jl operations ################################

@non_differentiable calculate_sectors(total_size, even_size)
@non_differentiable calculate_fused_size(total_size, even_size)
@non_differentiable prepare_fused_info(total_size, even_size, index_types, index_type_fused, inds)

function ChainRulesCore.rrule(
    ::typeof(fuse),
    tensor::Grassmann{T, N1, AT},
    inds::NTuple{N2, Int};
    index_type_fused::Symbol=:in,
) where {T, N1, N2, AT}
    y = fuse(tensor, inds; index_type_fused=index_type_fused)
    min_ind = minimum(inds)
    total_size_in = size(tensor)
    even_size_in = even(tensor)
    index_type_in = index_type(tensor)

    function fuse_pullback(Δy)
        Δy_unthunk = unthunk(Δy)
        if Δy_unthunk isa AbstractZero
            return (NoTangent(), ZeroTangent(), NoTangent())
        end
        Δtensor = split(Δy_unthunk, min_ind, total_size_in, even_size_in, index_type_in)
        return (NoTangent(), Δtensor, NoTangent())
    end

    return y, fuse_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Base.split),
    tensor::Grassmann{T, N1, AT},
    ind::Int,
    total_size_split::NTuple{N2, Int},
    even_size_split::NTuple{N2, Int},
    index_type_split::NTuple{N2, Symbol},
) where {T, N1, N2, AT}
    y = split(tensor, ind, total_size_split, even_size_split, index_type_split)
    N = N2 - N1
    inds_split = ntuple(i -> i - 1 + ind, N + 1)
    index_type_fused = index_type(tensor)[ind]

    function split_pullback(Δy)
        Δy_unthunk = unthunk(Δy)
        if Δy_unthunk isa AbstractZero
            return (NoTangent(), ZeroTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        end
        Δtensor = fuse(Δy_unthunk, inds_split; index_type_fused=index_type_fused)
        return (NoTangent(), Δtensor, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return y, split_pullback
end
