################################# Grassmann iPEPS #################################

"""
Grassmann iPEPS tensors on the square lattice

The bond weights {Λ} are stored in the Λx and Λy matrices, which may be absent (i.e. set to Missing)

Index reading order: Phys, Left, Right, Up, Down
Corresponding arrow: :out, :out, :in, :in, :out

                      ↑     
                      ↑           ↙
                      ↑    Λx[x-1, y]
                      ↑     ↙
                      ↑  ↙
⟵⟵ Λy[x, y] ⟵⟵ A[x, y] ⟵⟵ Λy[x, y+1] ⟵⟵
                   ↙   
                 ↙    
           Λx[x, y]   
            ↙ 
"""

struct Square_GPEPS{T<:Number}
    A::Matrix{Grassmann{T, 5}}
    Λx::Union{Missing, Matrix{GrassmannMatrix{Float64}}}
    Λy::Union{Missing, Matrix{GrassmannMatrix{Float64}}}
end

function Square_GPEPS(
    Dphys::Int, Dphys_even::Int, Dvir::Int, 
    Lx::Int, Ly::Int, Q::Type, has_bond_weights::Bool)

    A = Matrix{Grassmann{Q, 5}}(undef, Lx, Ly)

    Dvir_even = Int(Dvir/2)

    for row in 1:Lx, col in 1:Ly
        A[row, col] = Grassmann(
            (Dphys, Dvir, Dvir, Dvir, Dvir), 
            (Dphys_even, Dvir_even, Dvir_even, Dvir_even, Dvir_even), 
            (:out, :out, :in, :in, :out), Q; init=:random)
    end

    if has_bond_weights
        Λx = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)
        Λy = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)
        for row in 1:Lx, col in 1:Ly
            Λx[row, col] = Grassmann(prepare_bond_weight(Dvir, Dvir_even), (Dvir, Dvir), (Dvir_even, Dvir_even), (:out, :in))
            Λy[row, col] = Grassmann(prepare_bond_weight(Dvir, Dvir_even), (Dvir, Dvir), (Dvir_even, Dvir_even), (:out, :in))
        end
    else
        Λx = missing
        Λy = missing
    end

    return Square_GPEPS{Q}(A, Λx, Λy)
end

Base.size(peps::Square_GPEPS) = size(peps.A)
Base.eltype(peps::Square_GPEPS{Q}) where {Q} = Q

function absorb_Schmidt_weights(peps::Square_GPEPS{T}) where {T}

    Lx, Ly = size(peps)

    for r in 1:Lx, c in 1:Ly
        r_m1 = Nmod(r-1, Lx)
        c_p1 = Nmod(c+1, Ly)
        # Ao1[phy, r, u, d, l] = peps.A[r, c][phy, dum, r, u, d] * sqrt(peps.Λy[r, c])[l, dum]
        Ao1 = contract(peps.A[r, c], sqrt(peps.Λy[r, c]), (2, 2); sign_function=global_sign)
        # Ao2[phy, u, d, l, r] = Ao1[phy, dum, u, d, l] * sqrt(peps.Λy[r, c_p1])[dum, r]
        Ao2 = contract(Ao1, sqrt(peps.Λy[r, c_p1]), (2, 1); sign_function=global_sign)
        # Ao3[phy, d, l, r, u] = Ao2[phy, dum, d, l, r] * sqrt(peps.Λx[r_m1, c])[dum, u]
        Ao3 = contract(Ao2, sqrt(peps.Λx[r_m1, c]), (2, 1); sign_function=global_sign)
        # peps.A[r, c][phy, l, r, u, d] = Ao3[phy, dum, l, r, u] * sqrt(peps.Λx[r, c])[d, dum]
        peps.A[r, c] = contract(Ao3, sqrt(peps.Λx[r, c]), (2, 2); sign_function=global_sign)
    end

    return Square_GPEPS{T}(peps.A, missing, missing)
end

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
