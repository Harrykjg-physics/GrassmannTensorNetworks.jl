################################# Imaginary time evolution with Nearest Neighbour (NN) Simple Update (SU)  #################################

function Grassmann_SU(
    Gate::Grassmann{Float64, 4}, 
    peps::Square_GPEPS{T}, 
    δτ::Float64, 
    Dbond::Int; 
    su_iter::Int=1000, 
    su_tol::Float64=1e-12, 
    average_trunc::Bool=false, 
    save_iter::Int=0, 
    start::Int=0) where {T}

    Lx, Ly = size(peps)

    Eavg = 1.0
    Λx_tmp = copy(peps.Λx)
    Λy_tmp = copy(peps.Λy)

    for i = start+1:start+su_iter

        ti = time()
        # Update the x bond
        peps, coefx = update_x_bond(Gate, peps, Dbond; average_trunc=average_trunc)
        # Update the y bond
        peps, coefy = update_y_bond(Gate, peps, Dbond; average_trunc=average_trunc)
        tf = time()

        coef = prod(coefx) * prod(coefy)
        Eavg_tmp = -(1/δτ) * (1/(Lx * Ly)) * log(coef) 
        conv_err_energy = (abs(Eavg_tmp) - abs(Eavg))/abs(Eavg)   

        # The convergence in the Schimidt spectrum
        Λx_diff = compare_weights(Λx_tmp, peps.Λx)
        Λy_diff = compare_weights(Λy_tmp, peps.Λy)
        Λ_diff_max = maximum([maximum(Λx_diff), maximum(Λy_diff)])

        @info @sprintf "Simple update iteration : %i,   δτ : %f,   Δt : %.5f" i  δτ  (tf-ti)
        @info @sprintf "---- Estimated ground state energy per site: %.9f"  Eavg_tmp
        @info @sprintf "---- Energy convergence: %.3e   Schmidt weight convergence: %.3e" conv_err_energy  Λ_diff_max

        if save_iter > 0 && mod(i, save_iter) == 0
            save_str = "iter$i"*"_δτ$δτ"
            save(peps, "tensor_file", save_str)
            save("su", save_str, "energy", Eavg_tmp, "err1", Λ_diff_max, "err2", conv_err_energy)
        end
        
        if Λ_diff_max < su_tol
            save_str = "iter$i"*"_δτ$δτ"
            save(peps, "tensor_file", save_str)
            save("su", save_str, "energy", Eavg_tmp, "err1", Λ_diff_max, "err2", conv_err_energy)
            break
        else
            Eavg = copy(Eavg_tmp)
            Λx_tmp = copy(peps.Λx)
            Λy_tmp = copy(peps.Λy)
        end
    end

    return peps
end

################################# Core functions for Nearest Neighbour GSU  #################################

"""
                                                         ↙
                                          ↑           ↙
                                          ↑    Λx[x-1, y]
                                          ↑     ↙
                                          ↑  ↙
                    ⟵⟵ Λy[x, y] ⟵⟵ A[x, y] ⟵⟵ Λy[x, y+1] ⟵⟵
                                       ↙   
                         ↑           ↙    
                         ↑     Λx[x, y]   
                         ↑     ↙ 
                         ↑  ↙
⟵⟵ Λy[x+1, y] ⟵⟵ A[x+1, y] ⟵⟵ Λy[x+1, y+1] ⟵⟵
                      ↙
                   ↙
             Λx[x+1, y]
             ↙
          ↙
"""

function update_x_bond(
    Gate::Grassmann{Float64, 4}, 
    peps::Square_GPEPS{T}, 
    Dcut::Int; 
    average_trunc::Bool=true) where {T}
  
    Lx, Ly = size(peps)
    coef_mat = Matrix{Float64}(undef, Lx, Ly)

    for r in 1:Lx, c in 1:Ly
        r_p1 = Nmod(r+1, Lx)
        r_m1 = Nmod(r-1, Lx)
        c_p1 = Nmod(c+1, Ly)
        c_m1 = Nmod(c-1, Ly)
        peps.A[r, c], peps.A[r_p1, c], peps.Λx[r, c], coef_mat[r, c] = update_x_bond(
          Gate, peps.A[r, c], peps.A[r_p1, c], 
          peps.Λx[r, c], peps.Λx[r_m1, c], peps.Λy[r, c], peps.Λy[r, c_p1], 
          peps.Λx[r_p1, c], peps.Λy[r_p1, c], peps.Λy[r_p1, c_p1], Dcut; 
          average_trunc=average_trunc)
    end

    return peps, coef_mat
end

"""

                                  ↙                             ↙
                      ↑         ↙                   ↑         ↙
                      ↑   Λx[x-1, y]                ↑  Λx[x-1, y+1]               
                      ↑    ↙                        ↑    ↙
                      ↑  ↙                          ↑  ↙
⟵⟵ Λy[x, y] ⟵⟵ A[x, y] ⟵⟵ Λy[x, y+1] ⟵⟵ A[x, y+1] ⟵⟵ Λy[x, y+2] ⟵⟵
                   ↙                              ↙
                 ↙                              ↙ 
            Λx[x, y]                       Λx[x, y+1]
           ↙                              ↙
         ↙                             ↙

To update the y bond using update_x_bond() function :
    a. Rotate the above configuration by 90 degrees anti-clockwisely around the physical bond.
    b. Switch the two virtual indices along the x direction.
    

                                                         ↙
                                          ↑           ↙
                                          ↑    Λy[x, y+2]
                                          ↑     ↙
                                          ↑  ↙
                 ⟵⟵ Λy[x, y+1] ⟵⟵ A[x, y+1] ⟵⟵ Λx[x-1, y+1] ⟵⟵
                                       ↙   
                         ↑           ↙    
                         ↑     Λy[x, y+1]   
                         ↑     ↙ 
                         ↑  ↙
  ⟵⟵ Λx[x, y] ⟵⟵ A[x, y] ⟵⟵ Λx[x-1, y] ⟵⟵
                      ↙
                   ↙
             Λy[x, y]
             ↙
          ↙

"""

function update_y_bond(
    Gate::Grassmann{Float64, 4}, 
    peps::Square_GPEPS{T}, 
    Dcut::Int; 
    average_trunc::Bool=true) where {T}
  
    Lx, Ly = size(peps)
    coef_mat = Matrix{Float64}(undef, Lx, Ly)

    for r in 1:Lx, c in 1:Ly
        r_m1 = Nmod(r-1, Lx)
        c_p1 = Nmod(c+1, Ly)
        c_p2 = Nmod(c+2, Ly)
        A_new_r_c_p1, A_new_r_c, peps.Λy[r, c_p1], coef_mat[r, c_p1] = update_x_bond(
          Gate, dirx2y(peps.A[r, c_p1]), dirx2y(peps.A[r, c]), 
          peps.Λy[r, c_p1], peps.Λy[r, c_p2], peps.Λx[r, c_p1], peps.Λx[r_m1, c_p1], 
          peps.Λy[r, c], peps.Λx[r, c], peps.Λx[r_m1, c], Dcut; 
          average_trunc=average_trunc)
        peps.A[r, c] = diry2x(A_new_r_c)
        peps.A[r, c_p1] = diry2x(A_new_r_c_p1)
    end

    return peps, coef_mat
end



"""
Step 1: Absorb the Schmidt weights :

                                                ↙
                                    ↑         ↙
                                    ↑     Λx1b
                                    ↑    ↙
                                    ↑  ↙
                    ⟵⟵ Λy1a ⟵⟵ A1 ⟵⟵ Λy1b ⟵⟵
                                  ↙   
                                ↙   
                            √Λx   
                 ↑        ↙
                 ↑     √Λx   
                 ↑    ↙ 
                 ↑  ↙
 ⟵⟵ Λy2a ⟵⟵ A2 ⟵⟵ Λy2b ⟵⟵
               ↙
             ↙
          Λx2a
         ↙
       ↙

Step 2: QR or LQ decompositions :


                               
                                                  ↙
              ↑      ↙                         ↙                           
              ↑    ↙                         ↙ 
              ↑  ↙                ⟵⟵⟵ Q1 ⟵⟵⟵
      ⟵⟵⟵ B1 ⟵⟵⟵  ===>        ↑    ↙
            ↙                       ↑  ↙
          ↙                         X1 
                                   ↙
                                 ↙

                                   
                                              ↑    ↙
              ↑      ↙                        ↑ ↙
              ↑    ↙                         X2
              ↑  ↙                          ↙
      ⟵⟵⟵ B2 ⟵⟵⟵  ===>              ↙
            ↙                  ⟵⟵⟵ Q2 ⟵⟵⟵
          ↙                           ↙    
        ↙                           ↙ 


Step 3 : Apply the Gate = exp(- δτ H) and perform SVD :
 
                                                      ↑
                                                      ↑
                      ↑                               ↑   ↙    
                      ↑ #                             ↑ ↙
                      #                               V†x
                ↑   # ↑    ↙                       ↙
                ↑ #   ↑  ↙               ↑       ↙
                #     X1                 ↑    Λx_new
              # ↑    ↙         ====>     ↑   ↙
                ↑ ↙                      ↑ ↙
               X2                        Ux
             ↙                         ↙
           ↙                         ↙


Step 4 : Recover the GPEPS tensor
                                                 ↙
              ↑      ↙                         ↙                           
              ↑    ↙                         ↙ 
              ↑  ↙                ⟵⟵⟵ R1 ⟵⟵⟵
      ⟵⟵⟵ A1o ⟵⟵⟵  <==        ↑    ↙
            ↙                       ↑  ↙
          ↙                         V†x 
                                   ↙
                                 ↙
       
                                              ↑    ↙
              ↑      ↙                        ↑ ↙
              ↑    ↙                         Ux
              ↑  ↙                          ↙
      ⟵⟵⟵ A2o ⟵⟵⟵  <==              ↙
            ↙                  ⟵⟵⟵ L2 ⟵⟵⟵
          ↙                           ↙    
        ↙                           ↙ 

Step 4 : Split the Schmidt weights and update the local tensors :

                                    ↑         ↙
                                    ↑   (Λx1b)⁻¹
                                    ↑    ↙
                                    ↑  ↙
                ⟵⟵ (Λy1a)⁻¹ ⟵⟵ A1o ⟵⟵ (Λy1b)⁻¹ ⟵⟵
                                  ↙   
                                ↙    
                     ↑        ↙
                     ↑     Λx_new   
                     ↑    ↙ 
                     ↑  ↙
 ⟵⟵ (Λy2a)⁻¹ ⟵⟵ A2o ⟵⟵ (Λy2b)⁻¹ ⟵⟵
                  ↙
                ↙
            (Λx2a)⁻¹
            ↙
          ↙                                                
"""

function update_x_bond(
    G::Grassmann{Float64, 4}, 
    A1::Grassmann{T, 5}, 
    A2::Grassmann{T, 5}, 
    Λx::Grassmann{Float64, 2}, 
    Λx1b::Grassmann{Float64, 2}, 
    Λy1a::Grassmann{Float64, 2}, 
    Λy1b::Grassmann{Float64, 2}, 
    Λx2a::Grassmann{Float64, 2}, 
    Λy2a::Grassmann{Float64, 2}, 
    Λy2b::Grassmann{Float64, 2}, 
    Dcut::Int; 
    average_trunc::Bool=true) where {T}

    # Absorb the Schmidt weights into the local tensor A1 and A2 at first
    # A1_tmp1[n1p, y1, x1, x1p, y1p] = A1[n1p, dum, y1, x1, x1p] * Λy1a[y1p, dum]
    A1_tmp1 = contract(A1, Λy1a, (2, 2); sign_function=global_sign)
    # A1_tmp2[n1p, x1, x1p, y1p, y1] = A1_tmp1[n1p, dum, x1, x1p, y1p] * Λy1b[dum, y1]
    A1_tmp2 = contract(A1_tmp1, Λy1b, (2, 1); sign_function=global_sign)
    # A1_tmp3[n1p, x1p, y1p, y1, x1] = A1_tmp2[n1p, dum, x1p, y1p, y1] * Λx1b[dum, x1]
    A1_tmp3 = contract(A1_tmp2, Λx1b, (2, 1); sign_function=global_sign)
    # B1[n1p, y1p, y1, x1, x1p] = A1_tmp3[n1p, dum, y1p, y1, x1] * √Λx[x1p, dum]
    B1 = contract(A1_tmp3, sqrt(Λx), (2, 2); sign_function=global_sign)

    # A2_tmp1[n2p, y2, x2, x2p, y2p] = A2[n2p, dum, y2, x2, x2p] * Λy2a[y2p, dum]
    A2_tmp1 = contract(A2, Λy2a, (2, 2); sign_function=global_sign)
    # A2_tmp2[n2p, x2, x2p, y2p, y2] = A2_tmp1[n2p, dum, x2, x2p, y2p] * Λy2b[dum, y2]
    A2_tmp2 = contract(A2_tmp1, Λy2b, (2, 1); sign_function=global_sign)
    # A2_tmp3[n2p, x2p, y2p, y2, x2] = A2_tmp2[n2p, dum, x2p, y2p, y2] * √Λx[dum, x2]
    A2_tmp3 = contract(A2_tmp2, sqrt(Λx), (2, 1); sign_function=global_sign)
    # B2[n2p, y2p, y2, x2, x2p] = A2_tmp3[n2p, dum, y2p, y2, x2] * Λx2a[x2p, dum]
    B2 = contract(A2_tmp3, Λx2a, (2, 2); sign_function=global_sign)

    # Bond projection technique
    # B1_perm[n1p, x1p, y1p, y1, x1] <-- B1[n1p, y1p, y1, x1, x1p]
    B1_perm = permutedims(B1, (1, 5, 2, 3, 4); sign_function=global_sign)
    # B1_perm[(n1p, x1p), (y1p, y1, x1)] --> X1[(n1p, x1p), x1], Q1[x1p, (y1p, y1, x1)]
    X1, Q1 = gortho(B1_perm, (1, 2), (3, 4, 5); alg=LinearAlgebra.lq)
    # B2_perm[y2p, y2, x2p, n2p, x2] <-- B2[n2p, y2p, y2, x2, x2p]
    B2_perm = permutedims(B2, (2, 3, 5, 1, 4); sign_function=global_sign)
    # B2_perm[(y2p, y2, x2p), (n2p, x2)] --> Q2[(y2p, y2, x2p), x2], X2[x2p, (n2p, x2)]
    Q2, X2 = gortho(B2_perm, (1, 2, 3), (4, 5); alg=LinearAlgebra.qr)

    # Apply the evolution gate G
    # C1[n2p, n1p, n2, x1p, x1] = G[n2p, n1p, n2, dum] * X1[(dum, x1p), x1]
    C1 = contract(G, X1, (4, 1); sign_function=global_sign)
    # C2[n2p, x2p, n1p, x1] <-- C2[n2p, n1p, x1, x2p] = C1[n2p, n1p, dum1, dum2, x1] * X2[x2p, (dum1, dum2)]
    C2 = contract(C1, X2, ((3, 4), (2, 3)); perm=(1, 4, 2, 3), sign_function=global_sign)

    # Grassmann SVD
    # C2[(n2p, x2p), (n1p, x1)] --> Ux[(n2p, x2p), x2], Λx_new[x2p, x1], Vxdag[x1p, (n1p, x1)] --> Vx[(n1, x1p), x1]  
    Ux, Λx_new, Vx, trunc_err = gsvd(C2, (1, 2), (3, 4), Dcut; trunc=true, average_trunc=average_trunc, sign_function=global_sign)

    # Split the Schmidt weights from the local tensors Ao1 and Bo1
    # A2o[n2p, x2, y2p, y2, x2p] = Ux[(n2p, dum), x2] * Q2[(y2p, y2, x2p), dum]
    A2o = contract(Ux, Q2, (2, 4); sign_function=global_sign)
    # A2o1[n2p, x2, y2, x2p, y2p] = A2o[n2p, x2, dum, y2, x2p] * inv(Λy2a)[y2p, dum]
    A2o1 = contract(A2o, inv(Λy2a), (3, 2); sign_function=global_sign)
    # A2o2[n2p, x2, x2p, y2p, y2] = A2o1[n2p, x2, dum, x2p, y2p] * inv(Λy2b)[dum, y2]
    A2o2 = contract(A2o1, inv(Λy2b), (3, 1); sign_function=global_sign)
    # A2o3[n2p, y2p, y2, x2, x2p] <-- A2o3[n2p, x2, y2p, y2, x2p] = A2o2[n2p, x2, dum, y2p, y2] * inv(Λx2a)[x2p, dum]
    A2o3 = contract(A2o2, inv(Λx2a), (3, 2); perm=(1, 3, 4, 2, 5), sign_function=global_sign)

    # A1o[n1p, x1p, y1p, y1, x1] = conj(Vx)[(n1p, dum), x1p] * Q1[dum, (y1p, y1, x1)]
    A1o = contract(Vx, Q1, (2, 1); cj=(true, false), sign_function=global_sign)
    # A1o1[n1p, x1p, y1, x1, y1p] = A1o[n1p, x1p, dum, y1, x1] * inv(Λy1a)[y1p, dum]
    A1o1 = contract(A1o, inv(Λy1a), (3, 2); sign_function=global_sign)
    # A1o2[n1p, x1p, x1, y1p, y1] = A1o1[n1p, x1p, dum, x1, y1p] * inv(Λy1b)[dum, y1]
    A1o2 = contract(A1o1, inv(Λy1b), (3, 1); sign_function=global_sign)
    # A1o3[n1p, y1p, y1, x1, x1p] <-- A1o3[n1p, x1p, y1p, y1, x1] = A1o2[n1p, x1p, dum, y1p, y1] * inv(Λx1b)[dum, x1]
    A1o3 = contract(A1o2, inv(Λx1b), (3, 1); perm=(1, 3, 4, 5, 2), sign_function=global_sign)

    coef = maximum(Λx_new)
    Λx_new = Λx_new/coef

    return A1o3, A2o3, Λx_new, coef
end

########################################### Square GPEPS tensor permutation ###########################################

function dirx2y(A::Grassmann{T, 5}) where {T}

  # A_perm[np, b, bp, a, ap] <-- A[np, ap, a, b, bp] 
  A_perm = permutedims(A, (1, 4, 5, 3, 2); sign_function=global_sign)
  # A_out[np, bp, b, a, ap] <-- A_perm[np, b, bp, a, ap]
  A_out = permutedims(A_perm, (1, 3, 2, 4, 5); sign_function=global_sign)
end

function diry2x(A_out::Grassmann{T, 5}) where {T}

  # A_perm[np, b, bp, a, ap] <-- A_out[np, bp, b, a, ap]
  A_perm = permutedims(A_out, (1, 3, 2, 4, 5); sign_function=global_sign)
  # A[np, ap, a, b, bp] <-- A_perm[np, b, bp, a, ap]
  A = permutedims(A_perm, (1, 5, 4, 2, 3); sign_function=global_sign)
end
