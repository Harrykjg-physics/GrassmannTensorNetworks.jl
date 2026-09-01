
abstract type AbstractModel end

############################# Non-relativistic fermions #############################

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
    t, γ, λ = promote(t, γ, λ)
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
    H_coef[2, 2, 1, 1] = -γ; H_coef[1, 1, 2, 2] = -γ
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
2D interacting spinless fermion model on the square lattice

H = ∑_(⟨i,j⟩) [ -t (c†_{i} c_{j} + h.c.) ]
    - μ ∑_i c†_{i} c_{i}
    + V ∑_(⟨i,j⟩) c†_{i} c_{i} c†_{j} c_{j}

H_nn_bond = - t (c†i ⊗ cj + c†j ⊗ ci)
            - μ/4 (c†i ci ⊗ Ij + Ii ⊗ c†j cj)
            + V (c†i ci ⊗ c†j cj)
"""

struct InteractingSpinlessFermion{T<:Real} <: AbstractModel
    t::T
    μ::T
    V::T
end

function InteractingSpinlessFermion(t::Real, μ::Real, V::Real)
    t, μ, V = promote(t, μ, V)
    return InteractingSpinlessFermion{typeof(t)}(t, μ, V)
end

function n_site_Fock_basis(model::InteractingSpinlessFermion{T}) where {T}
    n_coef = zeros(T, (2, 2))
    n_coef[2, 2] = 1
    return n_coef
end

function n_site(model::InteractingSpinlessFermion)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (2, 2), (1, 1), (:out, :in))
end

function nn_bond_Fock_basis(model::InteractingSpinlessFermion{T}) where {T}

    t = model.t
    μ = model.μ
    V = model.V

    H_coef = zeros(T, (2, 2, 2, 2))
    # < 1ᵢ0ⱼ | c†i ⊗ cj | 0ᵢ1ⱼ > = 1; < 0ᵢ1ⱼ | c†j ⊗ ci | 1ᵢ0ⱼ > = 1
    H_coef[2, 1, 1, 2] = -t; H_coef[1, 2, 2, 1] = -t
    # < 1ᵢ0ⱼ | c†i ci ⊗ Ij | 1ᵢ0ⱼ > = 1; < 1ᵢ1ⱼ | c†i ci ⊗ Ij | 1ᵢ1ⱼ > = 1
    H_coef[2, 1, 2, 1] = -μ/4; H_coef[2, 2, 2, 2] = -μ/4
    # < 0ᵢ1ⱼ | Ii ⊗ c†j cj | 0ᵢ1ⱼ > = 1; < 1ᵢ1ⱼ | Ii ⊗ c†j cj | 1ᵢ1ⱼ > = 1
    H_coef[1, 2, 1, 2] = -μ/4; H_coef[2, 2, 2, 2] += -μ/4
    # < 1ᵢ1ⱼ | c†i ci ⊗ c†j cj | 1ᵢ1ⱼ > = 1
    H_coef[2, 2, 2, 2] += V

    return H_coef
end

function nn_bond(model::InteractingSpinlessFermion)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::InteractingSpinlessFermion, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (4, 4))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (2, 2, 2, 2))
    G = Grassmann(G_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end

"""
2D t-J model on the square lattice

H = -t ∑_(⟨i,j⟩,σ) (c̃†_{iσ} c̃_{jσ} + h.c.)
    + J ∑_(⟨i,j⟩) (S_i ⋅ S_j - 1/4 n_i n_j)
    - μ ∑_i n_i

The local basis is (0, ↑, ↓), with double occupancy projected out.

H_nn_bond = -t projected hopping
            + J (S_i ⋅ S_j - 1/4 n_i n_j)
            - μ/4 (n_i ⊗ Ij + Ii ⊗ n_j)
"""

struct TJModel{T<:Real} <: AbstractModel
    t::T
    J::T
    μ::T
end

function TJModel(t::Real, J::Real, μ::Real)
    t, J, μ = promote(t, J, μ)
    return TJModel{typeof(t)}(t, J, μ)
end

function n_site_Fock_basis(model::TJModel{T}) where {T}
    n_coef = zeros(T, (3, 3))
    n_coef[2, 2] = 1
    n_coef[3, 3] = 1
    return n_coef
end

function n_site(model::TJModel)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (3, 3), (1, 1), (:out, :in))
end

function nn_bond_Fock_basis(model::TJModel{T}) where {T}

    t = model.t
    J = model.J
    μ = model.μ

    H_coef = zeros(T, (3, 3, 3, 3))
    # < ↑ᵢ0ⱼ | c̃†i↑ ⊗ c̃j↑ | 0ᵢ↑ⱼ > = 1; < 0ᵢ↑ⱼ | c̃†j↑ ⊗ c̃i↑ | ↑ᵢ0ⱼ > = 1
    H_coef[2, 1, 1, 2] = -t; H_coef[1, 2, 2, 1] = -t
    # < ↓ᵢ0ⱼ | c̃†i↓ ⊗ c̃j↓ | 0ᵢ↓ⱼ > = 1; < 0ᵢ↓ⱼ | c̃†j↓ ⊗ c̃i↓ | ↓ᵢ0ⱼ > = 1
    H_coef[3, 1, 1, 3] = -t; H_coef[1, 3, 3, 1] = -t
    # Onsite chemical potential, split over the four square-lattice bonds touching each site.
    H_coef[2, 1, 2, 1] = -μ/4; H_coef[3, 1, 3, 1] = -μ/4
    H_coef[1, 2, 1, 2] = -μ/4; H_coef[1, 3, 1, 3] = -μ/4
    H_coef[2, 2, 2, 2] = -μ/2; H_coef[3, 3, 3, 3] = -μ/2
    H_coef[2, 3, 2, 3] = -μ/2; H_coef[3, 2, 3, 2] = -μ/2
    # S_i ⋅ S_j - 1/4 n_i n_j: only antiparallel spins have diagonal -1/2 and spin-flip +1/2.
    H_coef[2, 3, 2, 3] += -J/2; H_coef[3, 2, 3, 2] += -J/2
    H_coef[3, 2, 2, 3] += J/2; H_coef[2, 3, 3, 2] += J/2

    return H_coef
end

function nn_bond(model::TJModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (3, 3, 3, 3), (1, 1, 1, 1), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::TJModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (9, 9))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (3, 3, 3, 3))
    G = Grassmann(G_coef, (3, 3, 3, 3), (1, 1, 1, 1), (:out, :out, :in, :in))
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
    t, U, μ = promote(t, U, μ)
    return HubbardModel{typeof(t)}(t, U, μ)
end

function nu_site_Fock_basis(model::HubbardModel{T}) where {T}
    
    Nu_coef = zeros(T, (4, 4))
    # < D | c†↑ c↑ | D > = 1
    Nu_coef[2, 2] = 1
    # < ↑ | c†↑ c↑ | ↑ > = 1 
    Nu_coef[3, 3] = 1

    return Nu_coef
end

function nd_site_Fock_basis(model::HubbardModel{T}) where {T}
    
    Nd_coef = zeros(T, (4, 4))
    # < D | c†↓ c↓ | D > = 1
    Nd_coef[2, 2] = 1
    # < ↓ | c†↓ c↓ | ↓ > = 1 
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
    # < Dᵢ~ⱼ | ni↑ ⊗ Ij | Dᵢ~ⱼ > = 1
    H_coef[2, 1, 2, 1] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4 
    H_coef[2, 3, 2, 3] -= μ/4; H_coef[2, 4, 2, 4] -= μ/4
    # < ↑ᵢ~ⱼ | ni↑ ⊗ Ij | ↑ᵢ~ⱼ > = 1
    H_coef[3, 1, 3, 1] -= μ/4; H_coef[3, 2, 3, 2] -= μ/4 
    H_coef[3, 3, 3, 3] -= μ/4; H_coef[3, 4, 3, 4] -= μ/4 
    # < Dᵢ~ⱼ | ni↓ ⊗ Ij | Dᵢ~ⱼ > = 1
    H_coef[2, 1, 2, 1] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4
    H_coef[2, 3, 2, 3] -= μ/4; H_coef[2, 4, 2, 4] -= μ/4 
    # < ↓ᵢ~ⱼ | ni↓ ⊗ Ij | ↓ᵢ~ⱼ > = 1
    H_coef[4, 1, 4, 1] -= μ/4; H_coef[4, 2, 4, 2] -= μ/4
    H_coef[4, 3, 4, 3] -= μ/4; H_coef[4, 4, 4, 4] -= μ/4 
    # < ~ᵢDⱼ | Ii ⊗ nj↑ | ~ᵢDⱼ > = 1
    H_coef[1, 2, 1, 2] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4 
    H_coef[3, 2, 3, 2] -= μ/4; H_coef[4, 2, 4, 2] -= μ/4 
    # < ~ᵢ↑ⱼ | Ii ⊗ nj↑ | ~ᵢ↑ⱼ > = 1
    H_coef[1, 3, 1, 3] -= μ/4; H_coef[2, 3, 2, 3] -= μ/4 
    H_coef[3, 3, 3, 3] -= μ/4; H_coef[4, 3, 4, 3] -= μ/4 
    # < ~ᵢDⱼ | Ii ⊗ nj↓ | ~ᵢDⱼ > = 1
    H_coef[1, 2, 1, 2] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4
    H_coef[3, 2, 3, 2] -= μ/4; H_coef[4, 2, 4, 2] -= μ/4
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


############################# Relativistic fermions #############################

struct Gross_Neveu_Wilson_model{T<:Real} <: AbstractModel
    μ::T
    m::T
    g2::T
end

function Gross_Neveu_Wilson_model(μ::Real, m::Real, g2::Real)
    μ, m, g2 = promote(μ, m, g2)
    return Gross_Neveu_Wilson_model{typeof(μ)}(μ, m, g2)
end

function PartitionFunctionTensor(model::Gross_Neveu_Wilson_model{T}) where {T}

    # Ā[i1, i2, j1p, j2p]
    Ā = zeros(ComplexF64, (2, 2, 2, 2))

    for i1 in 0:1, i2 in 0:1, j1p in 0:1, j2p in 0:1

        if (i1, i2, j1p, j2p) == (1, 1, 0, 0)
            Ā[i1+1, i2+1, j1p+1, j2p+1] = - 1 + im
        elseif (i1, i2, j1p, j2p) == (1, 0, 1, 0)
            Ā[i1+1, i2+1, j1p+1, j2p+1] = - 1 - 1
        elseif (i1, i2, j1p, j2p) == (1, 0, 0, 1)
            Ā[i1+1, i2+1, j1p+1, j2p+1] = - 1 - im
        elseif (i1, i2, j1p, j2p) == (0, 1, 1, 0)
            Ā[i1+1, i2+1, j1p+1, j2p+1] = - im - 1
        elseif (i1, i2, j1p, j2p) == (0, 1, 0, 1)
            Ā[i1+1, i2+1, j1p+1, j2p+1] = - im - im
        elseif (i1, i2, j1p, j2p) == (0, 0, 1, 1)
            Ā[i1+1, i2+1, j1p+1, j2p+1] = 1 - im
        end

    end

    # A[j1, j2, i1p, i2p]
    A = zeros(ComplexF64, (2, 2, 2, 2))

    for j1 in 0:1, j2 in 0:1, i1p in 0:1, i2p in 0:1

        if (j1, j2, i1p, i2p) == (1, 1, 0, 0)
            A[j1+1, j2+1, i1p+1, i2p+1] = 1 + im
        elseif (j1, j2, i1p, i2p) == (1, 0, 1, 0)
            A[j1+1, j2+1, i1p+1, i2p+1] = 1 + 1
        elseif (j1, j2, i1p, i2p) == (1, 0, 0, 1)
            A[j1+1, j2+1, i1p+1, i2p+1] = 1 - im
        elseif (j1, j2, i1p, i2p) == (0, 1, 1, 0)
            A[j1+1, j2+1, i1p+1, i2p+1] = - im + 1
        elseif (j1, j2, i1p, i2p) == (0, 1, 0, 1)
            A[j1+1, j2+1, i1p+1, i2p+1] = - im - im
        elseif (j1, j2, i1p, i2p) == (0, 0, 1, 1)
            A[j1+1, j2+1, i1p+1, i2p+1] = - 1 - im
        end
    end

    T_coef = zeros(ComplexF64, (4, 4, 4, 4))

    for i1 in 0:1, j1 in 0:1, i2 in 0:1, j2 in 0:1
        for i1p in 0:1, j1p in 0:1, i2p in 0:1, j2p in 0:1

            I = f(i1, j1); J = f(i2, j2)
            Ip = f(i1p, j1p); Jp = f(i2p, j2p)

            sign1 = i1 * (j1 + j2 + i1p + i2p) + i2 * (j2 + i1p + i2p) + j1p * (i1p + i2p) + j2p * i2p + i1p + i2p
            sign2 = i2 - j2 + i2p -j2p
            sign3 = i1 + j1 + i2 + j2 + i1p + j1p + i2p + j2p
            sign4 = i1 + i2 + j2 + i1p
            sign5 = i2 + j2 + i2p + j2p

            T_coef[I, J, Ip, Jp] = (-1)^sign1 * exp(0.5*μ*sign2) * (1/sqrt(2))^sign3 * (
                ((m+2*r)^2 + 2*g2) * isequal(i1 + i2 + j1p + j2p, 0) * isequal(j1 + j2 + i1p + i2p, 0) -
                (m+2*r) * isequal(i1 + i2 + j1p + j2p, 1) * isequal(j1 + j2 + i1p + i2p, 1) - 
                (-1)^sign4 * (+im)^sign5 * (m+2*r) * isequal(i1 + i2 + j1p + j2p, 1) * isequal(j1 + j2 + i1p + i2p, 1) - 
                Ā[i1+1, i2+1, j1p+1, j2p+1] * A[j1+1, j2+1, i1p+1, i2p+1]
            )

        end
    end

    return T_coef
end