function Nmod(n::Int64, N::Int64)
    return mod(n-1, N) + 1
end

function compare_weights(
    Λ1::Matrix{GrassmannMatrix{Float64}}, 
    Λ2::Matrix{GrassmannMatrix{Float64}})

    size(Λ1) == size(Λ2) || throw(DimensionMismatch("The unit cell size of Λ1 and Λ2 should be the same !"))
    Lx, Ly = size(Λ1)
    conv_err_mat = Matrix{Float64}(undef, Lx, Ly)

    for r in 1:Lx, c in 1:Ly
        if (size(Λ1[r, c]) == size(Λ2[r, c])) && (even(Λ1[r, c]) == even(Λ2[r, c]))
            conv_err_mat[r, c] = maximum(abs(Λ1[r, c] - Λ2[r, c]))
        else
            conv_err_mat[r, c] = 1.0
        end
    end

    return conv_err_mat
end