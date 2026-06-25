"""
Arrow convention :

C1 ←←←← E3 ←←←← C2                  
↓       ↓       ↑
↓       ↓       ↑
E1 ←←←← T ←←←←  E2
↓       ↓       ↑
↓       ↓       ↑
C3 →→→→ E4 →→→→ C4
"""

struct CTMRGEnv{T<:Number}
    El::Matrix{Grassmann{T, 3}}
    Er::Matrix{Grassmann{T, 3}}
    Eu::Matrix{Grassmann{T, 3}}
    Ed::Matrix{Grassmann{T, 3}}
    Clu::Matrix{GrassmannMatrix{T}}
    Cru::Matrix{GrassmannMatrix{T}}
    Cld::Matrix{GrassmannMatrix{T}}
    Crd::Matrix{GrassmannMatrix{T}}
end

# All the bulk tensors (in a unit cell) are of the same size 
function CTMRGEnv(
    Dphys_hor::Int, 
    Dphys_hor_even::Int, 
    Dphys_vert::Int, 
    Dphys_vert_even::Int, 
    Dvir::Int, 
    Dvir_even::Int, 
    Lx::Int, 
    Ly::Int, 
    T::Type)

    El = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Er = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Eu = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Ed = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Clu = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))
    Cru = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))
    Cld = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))
    Crd = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))

    for x in 1:Lx, y in 1:Ly
        El[x, y] = Grassmann((Dvir, Dvir, Dphys_hor), (Dvir_even, Dvir_even, Dphys_hor_even), (:in, :out, :in), T; init=:random)
        Er[x, y] = Grassmann((Dvir, Dvir, Dphys_hor), (Dvir_even, Dvir_even, Dphys_hor_even), (:out, :in, :out), T; init=:random)
        Eu[x, y] = Grassmann((Dvir, Dvir, Dphys_vert), (Dvir_even, Dvir_even, Dphys_vert_even), (:out, :in, :out), T; init=:random)
        Ed[x, y] = Grassmann((Dvir, Dvir, Dphys_vert), (Dvir_even, Dvir_even, Dphys_vert_even), (:in, :out, :in), T; init=:random)
        Clu[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:in, :out), T; init=:random)
        Cru[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:out, :in), T; init=:random)
        Cld[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:out, :in), T; init=:random)
        Crd[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:in, :out), T; init=:random)
    end

    return CTMRGEnv{T}(El, Er, Eu, Ed, Clu, Cru, Cld, Crd)
end

# Each bulk tensors (in a unit cell) may not be of the same size 
function CTMRGEnv(
    T_mat::Matrix{Grassmann{T, 4}}, 
    Dvir::Int, 
    Dvir_even::Int) where {T}

    Lx, Ly = size(T_mat)

    El = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Er = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Eu = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Ed = Matrix{Grassmann{T, 3}}(undef, (Lx, Ly))
    Clu = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))
    Cru = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))
    Cld = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))
    Crd = Matrix{GrassmannMatrix{T}}(undef, (Lx, Ly))

    for x in 1:Lx, y in 1:Ly
        Dphys_left, Dphys_right, Dphys_up, Dphys_down = size(T_mat[x, y])
        Dphys_left_even, Dphys_right_even, Dphys_up_even, Dphys_down_even = even(T_mat[x, y])
        El[x, y] = Grassmann((Dvir, Dvir, Dphys_left), (Dvir_even, Dvir_even, Dphys_left_even), (:in, :out, :in), T; init=:random)
        Er[x, y] = Grassmann((Dvir, Dvir, Dphys_right), (Dvir_even, Dvir_even, Dphys_right_even), (:out, :in, :out), T; init=:random)
        Eu[x, y] = Grassmann((Dvir, Dvir, Dphys_up), (Dvir_even, Dvir_even, Dphys_up_even), (:out, :in, :out), T; init=:random)
        Ed[x, y] = Grassmann((Dvir, Dvir, Dphys_down), (Dvir_even, Dvir_even, Dphys_down_even), (:in, :out, :in), T; init=:random)
        Clu[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:in, :out), T; init=:random)
        Cru[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:out, :in), T; init=:random)
        Cld[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:out, :in), T; init=:random)
        Crd[x, y] = Grassmann((Dvir, Dvir), (Dvir_even, Dvir_even), (:in, :out), T; init=:random)
    end

    return CTMRGEnv{T}(El, Er, Eu, Ed, Clu, Cru, Cld, Crd)
end

Base.size(env::CTMRGEnv) = size(env.El)
Base.eltype(env::CTMRGEnv{Q}) where {Q} = Q
CTMRGEnv(Dphys, Dphys_even, Dvir, Dvir_even, Lx, Ly, T) = CTMRGEnv(Dphys, Dphys_even, Dphys, Dphys_even, Dvir, Dvir_even, Lx, Ly, T)

function _save_ctmrg_tensor_grid!(parent, tensor_name::AbstractString, tensor_grid)

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

function _load_ctmrg_tensor_grid(
    parent,
    tensor_name::AbstractString,
    indextype::NTuple{N, Symbol},
    ::Type{T},
    Lx::Int,
    Ly::Int) where {T, N}

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

function save(env::CTMRGEnv, filename::String, param_str::String)

    Lx, Ly = size(env)

    h5open("$filename.h5", "cw") do fid
        datakeys = keys(fid)
        param_str in datakeys && delete_object(fid, param_str)
        "(Lx, Ly)" in datakeys || (fid["(Lx, Ly)"] = [Lx, Ly])

        param_group = create_group(fid, param_str)
        _save_ctmrg_tensor_grid!(param_group, "El", env.El)
        _save_ctmrg_tensor_grid!(param_group, "Er", env.Er)
        _save_ctmrg_tensor_grid!(param_group, "Eu", env.Eu)
        _save_ctmrg_tensor_grid!(param_group, "Ed", env.Ed)
        _save_ctmrg_tensor_grid!(param_group, "Clu", env.Clu)
        _save_ctmrg_tensor_grid!(param_group, "Cru", env.Cru)
        _save_ctmrg_tensor_grid!(param_group, "Cld", env.Cld)
        _save_ctmrg_tensor_grid!(param_group, "Crd", env.Crd)
    end

    return nothing
end

function load(filename::String, param_str::String, ::Type{CTMRGEnv})

    h5open("$filename.h5", "r") do fid
        Lx, Ly = read(fid["(Lx, Ly)"])
        param_group = fid[param_str]
        T = eltype(read(param_group["El"]["(1, 1)"]["data"]))

        El = _load_ctmrg_tensor_grid(param_group, "El", (:in, :out, :in), T, Lx, Ly)
        Er = _load_ctmrg_tensor_grid(param_group, "Er", (:out, :in, :out), T, Lx, Ly)
        Eu = _load_ctmrg_tensor_grid(param_group, "Eu", (:out, :in, :out), T, Lx, Ly)
        Ed = _load_ctmrg_tensor_grid(param_group, "Ed", (:in, :out, :in), T, Lx, Ly)
        Clu = _load_ctmrg_tensor_grid(param_group, "Clu", (:in, :out), T, Lx, Ly)
        Cru = _load_ctmrg_tensor_grid(param_group, "Cru", (:out, :in), T, Lx, Ly)
        Cld = _load_ctmrg_tensor_grid(param_group, "Cld", (:out, :in), T, Lx, Ly)
        Crd = _load_ctmrg_tensor_grid(param_group, "Crd", (:in, :out), T, Lx, Ly)

        return CTMRGEnv{T}(El, Er, Eu, Ed, Clu, Cru, Cld, Crd)
    end
end

# Find the CTMRG environment tensors with the maximum iteration given χ
function find_maxiter(env_file, χ)

    prefix = "χ$(χ)iter"

    h5open(env_file*".h5", "r") do fid
        iter_vec = Int[]

        for key in keys(fid)
            key_str = String(key)
            startswith(key_str, prefix) || continue
            iter_str = key_str[nextind(key_str, lastindex(prefix)):end]
            push!(iter_vec, parse(Int, iter_str))
        end

        isempty(iter_vec) && throw(ArgumentError("No CTMRG environment saved for χ = $χ"))

        return "χ$χ"*"iter$(maximum(iter_vec))"
    end
end