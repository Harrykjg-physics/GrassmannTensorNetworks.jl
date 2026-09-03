
function run_nested_GCTMRG!(
    peps::Square_GPEPS,
    Os_mat::AbstractMatrix{<:Grassmann{Q, 2}}, 
    nested::NestedNetwork, 
    ctmrg_env::CTMRGEnv, 
    χ::Int; 
    ctmrg_iter::Int=100, 
    ctmrg_tol::Float64=1e-12, 
    start::Int=0, 
    average_trunc::Bool=false, 
    verbosity::Int=0, 
    save_iter::Int=0, 
    save_filename::String="ctmrg_env") where {Q}

    size(peps) == size(Os_mat) || throw(DimensionMismatch("peps and Os_mat sizes differ"))
    size(ctmrg_env) == size(nested) || throw(DimensionMismatch("nested environment and network sizes differ"))
 
    T_bulk = nested.network
    Lx, Ly = size(T_bulk)

    coef_iter = ones(Float64, 4, Lx, Ly)
    coef = similar(coef_iter)

    Λd_iter, Λu_iter, Λl_iter, Λr_iter = prepare_Λ(Lx, Ly, χ)

    expval_avg_tmp = one(Q)
    count = 0

    for iter = (start+1):(start+ctmrg_iter)

        ti = time()

        coef[1, :, :], trunc_err_d, Λd = down_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_d = maximum(trunc_err_d)
        max_Λ_err_d = maximum(compare_weights(Λd, Λd_iter))

        coef[2, :, :], trunc_err_u, Λu = up_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_u = maximum(trunc_err_u)
        max_Λ_err_u = maximum(compare_weights(Λu, Λu_iter))

        coef[3, :, :], trunc_err_l, Λl = left_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_l = maximum(trunc_err_l)
        max_Λ_err_l = maximum(compare_weights(Λl, Λl_iter))

        coef[4, :, :], trunc_err_r, Λr = right_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_r = maximum(trunc_err_r)
        max_Λ_err_r = maximum(compare_weights(Λr, Λr_iter))

        tf = time()

        if verbosity > 0

            @info @sprintf " Down move of CTMRG Iterations %i ==>  
            max_trunc_err_d :  %.6e   max_Λ_err_d : %.6e  " iter max_trunc_err_d max_Λ_err_d

            @info @sprintf " Up move of CTMRG Iterations %i ==>  
            max_trunc_err_u :  %.6e   max_Λ_err_u : %.6e  " iter max_trunc_err_u max_Λ_err_u

            @info @sprintf " Left move of CTMRG Iterations %i ==>  
            max_trunc_err_l :  %.6e   max_Λ_err_l : %.6e  " iter max_trunc_err_l max_Λ_err_l

            @info @sprintf " Right move of CTMRG Iterations %i ==>  
            max_trunc_err_r :  %.6e   max_Λ_err_r : %.6e  " iter max_trunc_err_r max_Λ_err_r
        end

        max_coef_err = maximum(abs.((coef - coef_iter)))/maximum(coef_iter)
        max_trunc_err = max(max(max_trunc_err_d, max_trunc_err_u), max(max_trunc_err_l, max_trunc_err_r))
        max_Λ_err = max(max(max_Λ_err_d, max_Λ_err_u), max(max_Λ_err_l, max_Λ_err_r))
        
        @info @sprintf "        "
        @info @sprintf " CTMRG Iterations %i ==>  Δt : %.4f   
        max_coef_err :  %.6e   max_trunc_err : %.6e  max_Λ_err : %.6e  " iter (tf-ti) max_coef_err max_trunc_err max_Λ_err
        @info @sprintf "        "

        # Whether to calculate and print the expectation value after each iteration
        if verbosity > 1  

            ti = time()
            _, expval = compute_nested_exp_site(nested, peps, Os_mat, ctmrg_env)
            expval_avg = sum(expval)/length(expval)
            tf = time()
            
            expval_diff = abs((expval_avg - expval_avg_tmp) / expval_avg_tmp)

            @info @sprintf " Expectation Calculation of CTMRG Iterations %i ==>  
            Δt : %.4f    Exp_avg : %.8f    Exp_diff : %.6e  " iter (tf-ti)  expval_avg expval_diff 
            @info @sprintf "        "

            # the convergence in the expectation value
            if (expval_diff < ctmrg_tol) && (count == 10)
                savestr = "χ$χ"*"iter$iter"
                save(ctmrg_env, save_filename, savestr)
                break
            elseif (expval_diff < ctmrg_tol) && (count < 10)
                expval_avg_tmp = copy(expval_avg)
                count += 1
            else
                expval_avg_tmp = copy(expval_avg)
            end
        end

        if save_iter > 0 && mod(iter, save_iter) == 0
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
        end

        if max_Λ_err < ctmrg_tol
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
            break
        else
            copyto!(coef_iter, coef)
            Λd_iter = copy(Λd)
            Λu_iter = copy(Λu)
            Λl_iter = copy(Λl)
            Λr_iter = copy(Λr)
        end 
    end
end

function run_nested_GCTMRG!(
    peps::Square_GPEPS,
    Ob_mat::AbstractMatrix{<:Grassmann{Q, 4}}, 
    nested::NestedNetwork, 
    ctmrg_env::CTMRGEnv, 
    χ::Int; 
    ctmrg_iter::Int=100, 
    ctmrg_tol::Float64=1e-12, 
    start::Int=0, 
    average_trunc::Bool=false, 
    verbosity::Int=0, 
    save_iter::Int=0, 
    save_filename::String="ctmrg_env") where {Q}

    size(peps) == size(Ob_mat) || throw(DimensionMismatch("peps and Ob_mat sizes differ"))
    size(ctmrg_env) == size(nested) || throw(DimensionMismatch("nested environment and network sizes differ"))
 
    T_bulk = nested.network
    Lx, Ly = size(T_bulk)

    coef_iter = ones(Float64, 4, Lx, Ly)
    coef = similar(coef_iter)

    Λd_iter, Λu_iter, Λl_iter, Λr_iter = prepare_Λ(Lx, Ly, χ)

    expval_avg_tmp = one(Q)
    count = 0

    for iter = (start+1):(start+ctmrg_iter)

        ti = time()

        coef[1, :, :], trunc_err_d, Λd = down_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_d = maximum(trunc_err_d)
        max_Λ_err_d = maximum(compare_weights(Λd, Λd_iter))

        coef[2, :, :], trunc_err_u, Λu = up_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_u = maximum(trunc_err_u)
        max_Λ_err_u = maximum(compare_weights(Λu, Λu_iter))

        coef[3, :, :], trunc_err_l, Λl = left_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_l = maximum(trunc_err_l)
        max_Λ_err_l = maximum(compare_weights(Λl, Λl_iter))

        coef[4, :, :], trunc_err_r, Λr = right_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_r = maximum(trunc_err_r)
        max_Λ_err_r = maximum(compare_weights(Λr, Λr_iter))

        tf = time()

        if verbosity > 0

            @info @sprintf " Down move of CTMRG Iterations %i ==>  
            max_trunc_err_d :  %.6e   max_Λ_err_d : %.6e  " iter max_trunc_err_d max_Λ_err_d

            @info @sprintf " Up move of CTMRG Iterations %i ==>  
            max_trunc_err_u :  %.6e   max_Λ_err_u : %.6e  " iter max_trunc_err_u max_Λ_err_u

            @info @sprintf " Left move of CTMRG Iterations %i ==>  
            max_trunc_err_l :  %.6e   max_Λ_err_l : %.6e  " iter max_trunc_err_l max_Λ_err_l

            @info @sprintf " Right move of CTMRG Iterations %i ==>  
            max_trunc_err_r :  %.6e   max_Λ_err_r : %.6e  " iter max_trunc_err_r max_Λ_err_r
        end

        max_coef_err = maximum(abs.((coef - coef_iter)))/maximum(coef_iter)
        max_trunc_err = max(max(max_trunc_err_d, max_trunc_err_u), max(max_trunc_err_l, max_trunc_err_r))
        max_Λ_err = max(max(max_Λ_err_d, max_Λ_err_u), max(max_Λ_err_l, max_Λ_err_r))
        
        @info @sprintf "        "
        @info @sprintf " CTMRG Iterations %i ==>  Δt : %.4f   
        max_coef_err :  %.6e   max_trunc_err : %.6e  max_Λ_err : %.6e  " iter (tf-ti) max_coef_err max_trunc_err max_Λ_err
        @info @sprintf "        "

        # Whether to calculate and print the expectation value after each iteration
        if verbosity > 1  

            ti = time()
            _, Eh = compute_nested_exp_hbond(nested, peps, Ob_mat, ctmrg_env)
            _, Ev = compute_nested_exp_vbond(nested, peps, Ob_mat, ctmrg_env)
            expval_avg = Es_avg = (sum(Eh) + sum(Ev)) / length(Eh)
            tf = time()
            
            expval_diff = abs((expval_avg - expval_avg_tmp) / expval_avg_tmp)

            @info @sprintf " Expectation Calculation of CTMRG Iterations %i ==>  
            Δt : %.4f    Exp_avg : %.8f    Exp_diff : %.6e  " iter (tf-ti)  expval_avg expval_diff 
            @info @sprintf "        "

            # the convergence in the expectation value
            if (expval_diff < ctmrg_tol) && (count == 10)
                savestr = "χ$χ"*"iter$iter"
                save(ctmrg_env, save_filename, savestr)
                break
            elseif (expval_diff < ctmrg_tol) && (count < 10)
                expval_avg_tmp = copy(expval_avg)
                count += 1
            else
                expval_avg_tmp = copy(expval_avg)
            end
        end

        if save_iter > 0 && mod(iter, save_iter) == 0
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
        end

        if max_Λ_err < ctmrg_tol
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
            break
        else
            copyto!(coef_iter, coef)
            Λd_iter = copy(Λd)
            Λu_iter = copy(Λu)
            Λl_iter = copy(Λl)
            Λr_iter = copy(Λr)
        end 
    end
end


