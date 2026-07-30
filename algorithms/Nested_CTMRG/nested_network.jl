struct NestedLayout
    source_size::Tuple{Int, Int}
    nested_size::Tuple{Int, Int}
    ket_sites::Matrix{CartesianIndex{2}}
    y_sites::Matrix{CartesianIndex{2}}
    x_sites::Matrix{CartesianIndex{2}}
    bra_sites::Matrix{CartesianIndex{2}}
end

function NestedLayout(source_size::Tuple{Int, Int})
    rows, cols = source_size
    rows > 0 && cols > 0 ||
        throw(ArgumentError("source unit-cell dimensions must be positive"))
    ket = [CartesianIndex(2r - 1, 2c - 1) for r in 1:rows, c in 1:cols]
    y = [CartesianIndex(2r - 1, 2c) for r in 1:rows, c in 1:cols]
    x = [CartesianIndex(2r, 2c - 1) for r in 1:rows, c in 1:cols]
    bra = [CartesianIndex(2r, 2c) for r in 1:rows, c in 1:cols]
    return NestedLayout(source_size, (2rows, 2cols), ket, y, x, bra)
end

NestedLayout(peps::Square_GPEPS) = NestedLayout(size(peps))
Base.size(layout::NestedLayout) = layout.nested_size
Base.size(layout::NestedLayout, dim::Integer) = layout.nested_size[dim]

struct NestedNetwork{T<:Number, X<:AbstractMatrix}
    network::Matrix{Grassmann{T, 4}}
    layout::NestedLayout
    x_crossings::X
end

Base.size(nested::NestedNetwork, args...) = size(nested.network, args...)
Base.axes(nested::NestedNetwork, args...) = axes(nested.network, args...)
Base.getindex(nested::NestedNetwork, inds...) = getindex(nested.network, inds...)
