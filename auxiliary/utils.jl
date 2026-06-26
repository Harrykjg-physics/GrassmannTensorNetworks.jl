function Nmod(n::Int64, N::Int64)
    return mod(n-1, N) + 1
end

# Compare two Schimidt weights and return the difference matrix
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

# Save arbitary number of results given the parameter string
function save(filename::String, param_str::String, args...)

    num = length(args)
    num > 0 && iseven(num) || throw(ArgumentError("args should be string-value pairs !"))
    
    fid = h5open("$filename.h5", "cw")
    param_str in keys(fid) ? delete_object(fid, "$param_str") : nothing
    create_group(fid, param_str)

    for i in 1:2:num
        fid[param_str][args[i]] = args[i+1]
    end

    close(fid)
end
