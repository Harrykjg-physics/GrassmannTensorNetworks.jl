
abstract type AbstractModel end

"""
2D Spinless Fermion model on the square lattice

H = ∑_(⟨i,j⟩) [ -t (c†_{i} c_{j} + h.c.)  - γ (c†_{i} c_{j}† + h.c.) ]
    - 2λ ∑_i c†_{i} c_{i}

H_nn_bond = - t ( c†i ⊗ cj + c†j ⊗ ci) 
            - γ ( c†i ⊗ c†j + cj ⊗ ci) 
            - λ/2 ( c†i ci ⊗ Ij + Ii ⊗ c†j cj) 
"""

struct SpinlessFermionModel{T<:Real} <: AbstractModel
    t::T
    γ::T
    λ::T
end

function SpinlessFermionModel(t::Real, γ::Real, λ::Real)
    return SpinlessFermionModel{typeof(t)}(t, γ, λ)
end

function n_site_Fock_basis(model::SpinlessFermionModel{T}) where {T}
    n_coef = zeros(T, (2, 2))
    n_coef[2, 2] = 1
    return n_coef
end

function n_site(model::SpinlessFermionModel)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (2, 2), (1, 1), (:out, :in))
end

function nn_bond_Fock_basis(model::SpinlessFermionModel{T}) where {T}

    t = model.t
    γ = model.γ
    λ = model.λ
    
    H_coef = zeros(T, (2, 2, 2, 2))
    # < 1ᵢ0ⱼ | c†i ⊗ cj | 0ᵢ1ⱼ > = 1; < 0ᵢ1ⱼ | c†j ⊗ ci | 1ᵢ0ⱼ > = 1
    H_coef[2, 1, 1, 2] = -t; H_coef[1, 2, 2, 1] = -t
    # < 1ᵢ1ⱼ | c†i ⊗ c†j | 0ᵢ0ⱼ > = 1; < 0ᵢ0ⱼ | cj ⊗ ci | 1ᵢ1ⱼ > = 1
    # TODO: Not sure about the sign here, but it's correct
    H_coef[2, 2, 1, 1] = γ; H_coef[1, 1, 2, 2] = γ
    # < 1ᵢ0ⱼ | c†i ci ⊗ Ij | 1ᵢ0ⱼ > = 1; < 1ᵢ1ⱼ | c†i ci ⊗ Ij | 1ᵢ1ⱼ > = 1
    H_coef[2, 1, 2, 1] = -λ/2; H_coef[2, 2, 2, 2] = -λ/2
    # < 0ᵢ1ⱼ | Ii ⊗ c†j cj | 0ᵢ1ⱼ > = 1; < 1ᵢ1ⱼ | Ii ⊗ c†j cj | 1ᵢ1ⱼ > = 1
    H_coef[1, 2, 1, 2] = -λ/2; H_coef[2, 2, 2, 2] += -λ/2

    return H_coef
end

function nn_bond(model::SpinlessFermionModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::SpinlessFermionModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (4, 4))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (2, 2, 2, 2))
    G = Grassmann(G_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end

"""
2D Fermi-Hubbard model on the square lattice

H = -t ∑_(⟨i,j⟩,σ) (c†_{iσ} c_{jσ} + h.c.) 
    + U ∑_i n_{i↑}n_{i↓} 
    - μ ∑_i(n_i↑ + n_i↓)

H_nn_bond = - t( c†i↑ ⊗ cj↑ + c†j↑ ⊗ ci↑  + c†i↓ ⊗ cj↓ + c†j↓ ⊗ ci↓) 
         + U/4 (ni↑ ni↓ ⊗ Ij + Ii ⊗ nj↑ nj↓) 
         - μ/4 (ni↑ ⊗ Ij + ni↓ ⊗ Ij + Ii ⊗ nj↑ + Ii ⊗ nj↓)
"""

struct HubbardModel{T<:Real} <: AbstractModel
    t::T
    U::T
    μ::T
end

function HubbardModel(t::Real, U::Real, μ::Real)
    return HubbardModel{typeof(t)}(t, U, μ)
end

function nu_site_Fock_basis(model::HubbardModel{T}) where {T}
    
    Nu_coef = zeros(T, (4, 4))
    # < ↑ | c†↑ c↑ | ↑ > = 1 
    Nu_coef[1, 1] = 1
    # < D | c†↑ c↑ | D > = 1
    Nu_coef[4, 4] = 1

    return Nu_coef
end

function nd_site_Fock_basis(model::HubbardModel{T}) where {T}
    
    Nd_coef = zeros(T, (4, 4))
    # < ↓ | c†↓ c↓ | ↓ > = 1 
    Nd_coef[2, 2] = 1
    # < D | c†↓ c↓ | D > = 1
    Nd_coef[4, 4] = 1

    return Nd_coef
end

function n_site_Fock_basis(model::HubbardModel{T}) where {T}
    nu_site_Fock_basis(model) + nd_site_Fock_basis(model)
end

function n_site(model::HubbardModel)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (4, 4), (2, 2), (:out, :in))
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
    H_coef[2, 1, 2, 1] += U/4; H_coef[2, 2, 2, 2] += U/4
    H_coef[2, 3, 2, 3] += U/4; H_coef[2, 4, 2, 4] += U/4
    # < ~ᵢDⱼ | Ii ⊗ nj↑ nj↓ | ~ᵢDⱼ > = 1
    H_coef[1, 2, 1, 2] += U/4; H_coef[2, 2, 2, 2] += U/4
    H_coef[3, 2, 3, 2] += U/4; H_coef[4, 2, 4, 2] += U/4
    # < ↑ᵢ~ⱼ | ni↑ ⊗ Ij | ↑ᵢ~ⱼ > = 1
    H_coef[3, 1, 3, 1] -= μ/4; H_coef[3, 2, 3, 2] -= μ/4 
    H_coef[3, 3, 3, 3] -= μ/4; H_coef[3, 4, 3, 4] -= μ/4 
    # < ↓ᵢ~ⱼ | ni↓ ⊗ Ij | ↓ᵢ~ⱼ > = 1
    H_coef[4, 1, 4, 1] -= μ/4; H_coef[4, 2, 4, 2] -= μ/4
    H_coef[4, 3, 4, 3] -= μ/4; H_coef[4, 4, 4, 4] -= μ/4 
    # < ~ᵢ↑ⱼ | Ii ⊗ nj↑ | ~ᵢ↑ⱼ > = 1
    H_coef[1, 3, 1, 3] -= μ/4; H_coef[2, 3, 2, 3] -= μ/4 
    H_coef[3, 3, 3, 3] -= μ/4; H_coef[4, 3, 4, 3] -= μ/4 
    # < ~ᵢ↓ⱼ | Ii ⊗ nj↓ | ~ᵢ↓ⱼ > = 1
    H_coef[1, 4, 1, 4] -= μ/4; H_coef[2, 4, 2, 4] -= μ/4
    H_coef[3, 4, 3, 4] -= μ/4; H_coef[4, 4, 4, 4] -= μ/4

    return H_coef
end

function nn_bond(model::HubbardModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (4, 4, 4, 4), (2, 2, 2, 2), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::HubbardModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (16, 16))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (4, 4, 4, 4))
    G = Grassmann(G_coef, (4, 4, 4, 4), (2, 2, 2, 2), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end
