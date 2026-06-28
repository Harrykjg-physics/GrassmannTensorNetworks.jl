
struct AbstractModel end

"""
2D Fermi-Hubbard model on the square lattice

H = -t ∑_(⟨i,j⟩,σ) (c†_{iσ} c_{jσ} + h.c.) 
    + U ∑_i n_{i↑}n_{i↓} 
    - μ ∑_i(n_i↑ + n_i↓)

H_nn_bond = - t( c†i↑ ⊗ cj↑ + c†j↑ ⊗ ci↑  + c†i↓ ⊗ cj↓ + c†j↓ ⊗ ci↓) 
         + U/2 (ni↑ ni↓ ⊗ Ij + Ii ⊗ nj↑ nj↓) 
         - μ/2 (ni↑ ⊗ Ij + ni↓ ⊗ Ij + Ii ⊗ nj↑ + Ii ⊗ nj↓)
"""

struct HubbardModel{T<:Real} <: AbstractModel
    t::T
    U::T
    μ::T
end

function HubbardModel(t::Real, U::Real, μ::Real)
    return HubbardModel{typeof(t)}(t, U, μ)
end

function nn_bond_Fock_basis(model::HubbardModel{T}) where {T}

    t = model.t
    U = model.U
    μ = model.μ
    
    H_coef = zeros(T, (4, 4, 4, 4))
    # < ↑ᵢ0ⱼ | c†i↑ ⊗ cj↑ | 0ᵢ↑ⱼ > = 1; < Dᵢ0ⱼ | c†i↑ ⊗ cj↑ | ↓ᵢ↑ⱼ > = -1
    # < ↑ᵢ↓ⱼ | c†i↑ ⊗ cj↑ | 0ᵢDⱼ > = 1; < Dᵢ↓ⱼ | c†i↑ ⊗ cj↑ | ↓ᵢDⱼ > = -1
    H_coef[3, 1, 1, 3] = -t; H_coef[2, 1, 4, 3] = t
    H_coef[3, 4, 1, 2] = -t; H_coef[2, 4, 4, 2] = t
    # < 0ᵢ↑ⱼ | c†j↑ ⊗ ci↑ | ↑ᵢ0ⱼ > = 1; < ↓ᵢ↑ⱼ | c†j↑ ⊗ ci↑ | Dᵢ0ⱼ > = -1
    # < 0ᵢDⱼ | c†j↑ ⊗ ci↑ | ↑ᵢ↓ⱼ > = 1; < ↓ᵢDⱼ | c†j↑ ⊗ ci↑ | Dᵢ↓ⱼ > = -1
    H_coef[1, 3, 3, 1] = -t; H_coef[4, 3, 2, 1] = t
    H_coef[1, 2, 3, 4] = -t; H_coef[4, 2, 2, 4] = t
    # < ↓ᵢ0ⱼ | c†i↓ ⊗ cj↓ | 0ᵢ↓ⱼ > = 1; < Dᵢ0ⱼ | c†i↓ ⊗ cj↓ | ↑ᵢ↓ⱼ > = 1
    # < ↓ᵢ↑ⱼ | c†i↓ ⊗ cj↓ | 0ᵢDⱼ > = -1; < Dᵢ↑ⱼ | c†i↓ ⊗ cj↓ | ↑ᵢDⱼ > = -1
    H_coef[4, 1, 1, 4] = -t; H_coef[2, 1, 3, 4] = -t
    H_coef[4, 3, 1, 2] = t; H_coef[2, 3, 3, 2] = t
    # < 0ᵢ↓ⱼ | c†j↓ ⊗ ci↓ | ↓ᵢ0ⱼ > = 1; < ↑ᵢ↓ⱼ | c†j↓ ⊗ ci↓ | Dᵢ0ⱼ > = 1
    # < 0ᵢDⱼ | c†j↓ ⊗ ci↓ | ↓ᵢ↑ⱼ > = -1; < ↑ᵢDⱼ | c†j↓ ⊗ ci↓ | Dᵢ↑ⱼ > = -1
    H_coef[1, 4, 4, 1] = -t; H_coef[3, 4, 2, 1] = -t
    H_coef[1, 2, 4, 3] = t; H_coef[3, 2, 2, 3] = t
    # < Dᵢ~ⱼ | ni↑ ni↓ ⊗ Ij | Dᵢ~ⱼ > = 1
    H_coef[2, 1, 2, 1] += U/2; H_coef[2, 2, 2, 2] += U/2
    H_coef[2, 3, 2, 3] += U/2; H_coef[2, 4, 2, 4] += U/2
    # < ~ᵢDⱼ | Ii ⊗ nj↑ nj↓ | ~ᵢDⱼ > = 1
    H_coef[1, 2, 1, 2] += U/2; H_coef[2, 2, 2, 2] += U/2
    H_coef[3, 2, 3, 2] += U/2; H_coef[4, 2, 4, 2] += U/2
    # < ↑ᵢ~ⱼ | ni↑ ⊗ Ij | ↑ᵢ~ⱼ > = 1
    H_coef[3, 1, 3, 1] -= μ/2; H_coef[3, 2, 3, 2] -= μ/2 
    H_coef[3, 3, 3, 3] -= μ/2; H_coef[3, 4, 3, 4] -= μ/2 
    # < ↓ᵢ~ⱼ | ni↓ ⊗ Ij | ↓ᵢ~ⱼ > = 1
    H_coef[4, 1, 4, 1] -= μ/2; H_coef[4, 2, 4, 2] -= μ/2
    H_coef[4, 3, 4, 3] -= μ/2; H_coef[4, 4, 4, 4] -= μ/2 
    # < ~ᵢ↑ⱼ | Ii ⊗ nj↑ | ~ᵢ↑ⱼ > = 1
    H_coef[1, 3, 1, 3] -= μ/2; H_coef[2, 3, 2, 3] -= μ/2 
    H_coef[3, 3, 3, 3] -= μ/2; H_coef[4, 3, 4, 3] -= μ/2 
    # < ~ᵢ↓ⱼ | Ii ⊗ nj↓ | ~ᵢ↓ⱼ > = 1
    H_coef[1, 4, 1, 4] -= μ/2; H_coef[2, 4, 2, 4] -= μ/2
    H_coef[3, 4, 3, 4] -= μ/2; H_coef[4, 4, 4, 4] -= μ/2

    return H_coef
end

function nn_bond(model::HubbardModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond = Grassmann(H_coef, (4, 4, 4, 4), (2, 2, 2, 2), (:out, :out, :in, :in))
    nn_bond_out = add_perm_sign(nn_bond, (1, 2, 4, 3))
end

function gate(model::HubbardModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (16, 16))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (4, 4, 4, 4))
    G = Grassmann(G_coef, (4, 4, 4, 4), (2, 2, 2, 2), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end
