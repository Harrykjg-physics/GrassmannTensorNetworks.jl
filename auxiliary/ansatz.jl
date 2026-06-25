################################# Grassmann iPEPS #################################

"""
Grassmann iPEPS tensors on the square lattice

The bond weights {Λ} are stored in the Λx and Λy matrices, which may be absent (i.e. set to Missing)

                      ↑     
                      ↑           ↙
                      ↑    Λy[x-1, y]
                      ↑     ↙
                      ↑  ↙
⟵⟵ Λx[x, y] ⟵⟵ A[x, y] ⟵⟵ Λx[x, y+1] ⟵⟵
                   ↙   
                 ↙    
           Λy[x, y]   
            ↙ 
"""

struct Square_GPEPS{T<:Number}
    A::Matrix{Grassmann{T, 5}}
    Λx::Union{Missing, Matrix{GrassmannMatrix{Float64}}}
    Λy::Union{Missing, Matrix{GrassmannMatrix{Float64}}}
end

Base.size(peps::Square_GPEPS) = size(peps.A)
Base.eltype(peps::Square_GPEPS{Q}) where {Q} = Q

function _save_square_gpeps_tensor_grid!(parent, tensor_name::AbstractString, tensor_grid)

    tensor_group = create_group(parent, tensor_name)

    @inbounds for row in axes(tensor_grid, 1), col in axes(tensor_grid, 2)
        site_group = create_group(tensor_group, "($row, $col)")
        tensor = tensor_grid[row, col]
        site_group["total_size"] = collect(size(tensor))
        site_group["even_size"] = collect(even(tensor))
        site_group["data"] = convert(Array, tensor)
    end

    return nothing
end

function _load_square_gpeps_tensor_grid(parent, tensor_name::AbstractString, indextype::NTuple{N, Symbol}, ::Type{T}, Lx::Int, Ly::Int) where {T, N}

    tensor_grid = Matrix{Grassmann{T, N}}(undef, Lx, Ly)
    tensor_group = parent[tensor_name]

    @inbounds for row in 1:Lx, col in 1:Ly
        site_group = tensor_group["($row, $col)"]
        total_size = Tuple(read(site_group["total_size"]))
        even_size = Tuple(read(site_group["even_size"]))
        data = read(site_group["data"])
        tensor_grid[row, col] = Grassmann(data, total_size, even_size, indextype)
    end

    return tensor_grid
end

# Save the square GPEPS tensors
function save(peps::Square_GPEPS, filename::String, param_str::String)

    Lx, Ly = size(peps)

    h5open("$filename.h5", "cw") do fid
        datakeys = keys(fid)
        param_str in datakeys && delete_object(fid, param_str)
        "(Lx, Ly)" in datakeys || (fid["(Lx, Ly)"] = [Lx, Ly])

        param_group = create_group(fid, param_str)
        _save_square_gpeps_tensor_grid!(param_group, "A", peps.A)
        peps.Λx !== missing && _save_square_gpeps_tensor_grid!(param_group, "Λx", peps.Λx)
        peps.Λy !== missing && _save_square_gpeps_tensor_grid!(param_group, "Λy", peps.Λy)
    end

    return nothing
end

# Load the square GPEPS tensors
function load(filename::String, param_str::String, ::Type{Square_GPEPS}; has_bond_weights::Bool=true)

    h5open("$filename.h5", "r") do fid
        Lx, Ly = read(fid["(Lx, Ly)"])
        param_group = fid[param_str]
        Q = eltype(read(param_group["A"]["(1, 1)"]["data"]))

        A = _load_square_gpeps_tensor_grid(param_group, "A", (:out, :out, :in, :in, :out), Q, Lx, Ly)

        if has_bond_weights
            Λx = _load_square_gpeps_tensor_grid(param_group, "Λx", (:out, :in), Float64, Lx, Ly)
            Λy = _load_square_gpeps_tensor_grid(param_group, "Λy", (:out, :in), Float64, Lx, Ly)
            return Square_GPEPS{Q}(A, Λx, Λy)
        else
            return Square_GPEPS{Q}(A, missing, missing)
        end
    end
end
