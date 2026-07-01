
########################################### Construct Reduced tensors for Square GPEPS ###########################################

""" 
                               [u']
                              ↗ 
                            ↗ 
                          ↗  
            [l] ⟶⟶⟶ Tup ⟶⟶⟶ [r']                                [U]
                      ↗ ↑                                            ↙
                    ↗   ↑                                          ↙
                  ↗     ↑                                        ↙
               [d]    [dum]     [u]      ===> [L'] ⟵⟵⟵ T_reduced ⟵⟵⟵ [R]
                        ↑      ↙                            ↙
                        ↑    ↙                            ↙
                        ↑  ↙                            ↙ 
           [l'] ⟵⟵⟵ Tdn ⟵⟵⟵ [r]                [D']
                      ↙
                    ↙  
                  ↙ 
                [d']  
"""

function reduced_tensor(Tdn::Grassmann{Q, 5}) where {Q}

    Tup = conj(Tdn; sign_function=global_sign)
    # T_reduced[l, l', r', r, u', u, d, d'] <-- T_reduced[l, r', u', d, l', r, u, d'] = Tup[dum, l, r', u', d] * Tdn[dum, l', r, u, d']
    T_reduced = contract(Tup, Tdn, (1, 1); perm=(1, 5, 2, 6, 3, 7, 4, 8), sign_function=global_sign)
    # T_reduced1[L', r', r, u', u, d, d'] <-- T_reduced[(l, l'), r', r, u', u, d, d']
    T_reduced = add_parity_sign(T_reduced, 1; sign_function=global_sign)
    T_reduced1 = fuse(T_reduced, (1, 2); index_type_fused=:out)
    # T_reduced2[L', R, u', u, d, d'] <-- T_reduced1[L', r', r, u', u, d, d']
    T_reduced1 = add_perm_sign(T_reduced1, (1, 3, 2, 4, 5, 6, 7); sign_function=global_sign)
    T_reduced2 = fuse(T_reduced1, (2, 3); index_type_fused=:in)
    # T_reduced3[L', R, U, d, d'] <-- T_reduced2[L', R, u', u, d, d']
    T_reduced2 = add_perm_sign(T_reduced2, (1, 2, 4, 3, 5, 6); sign_function=global_sign)
    T_reduced3 = fuse(T_reduced2, (3, 4); index_type_fused=:in)
    # T_reduced4[L', R, U, D'] <-- T_reduced3[L', R, U, d, d']
    T_reduced3 = add_parity_sign(T_reduced3, 4; sign_function=global_sign)
    T_reduced4 = fuse(T_reduced3, (4, 5); index_type_fused=:out)
end

function reduced_tensor(peps::Square_GPEPS{Q}) where {Q}

    Lx, Ly = size(peps)
  
    T_square_mat = Matrix{Grassmann{Q, 4}}(undef, Lx, Ly)

    for r in 1:Lx, c in 1:Ly
        T_square_mat[r, c] = reduced_tensor(peps.A[r, c])
    end

    return T_square_mat
end

""" 

Impurity bond along the y direction :

                               [u1']                             [u2']
                              ↗                                 ↗
                            ↗                                 ↗
                          ↗                                 ↗
         [l1] ⟶⟶⟶ Tup[x, y] ⟶⟶⟶ [dum1] ⟶⟶⟶ Tup[x, y+1] ⟶⟶⟶ [r2']                                        [U1]        [U2]
                      ↗ ↑                                ↗ ↑                                                       ↙           ↙
                    ↗ [dum3]       [u1]                ↗  [dum4]        [u2]                                     ↙           ↙ 
                  ↗     ↑          ↙                 ↗     ↑          ↙                                       ↙           ↙   
                ↗     #########################################      ↙           ====>   [L'] ⟵⟵⟵ ############################# ⟵⟵⟵ [R]
             [d1]       ↑      ↙                 ↗         ↑      ↙                                       ↙            ↙  
                      [dum5] ↙                [d2]       [dum6] ↙                                       ↙            ↙ 
                        ↑  ↙                               ↑ ↙                                        ↙            ↙
        [l1'] ⟵⟵⟵ Tdn[x, y] ⟵⟵⟵ [dum2] ⟵⟵⟵ Tdn[x, y+1] ⟵⟵⟵ [r2]                       [D1']          [D2'] 
                      ↙                                 ↙
                    ↙                                 ↙ 
                  ↙                                 ↙
              [d1']                               [d2']

"""  
     
function reduced_tensor_y_bond(Tdn1::Grassmann{Q, 5}, Tdn2::Grassmann{Q, 5}, H_bond::Grassmann{Q, 4}) where {Q}

    Tup1 = conj(Tdn1; sign_function=global_sign)
    Tup2 = conj(Tdn2; sign_function=global_sign)

    # Tdn[dum5, l1', u1, d1', dum6, r2, u2, d2'] = Tdn1[dum5, l1', dum2, u1, d1'] * Tdn2[dum6, dum2, r2, u2, d2']
    Tdn = contract(Tdn1, Tdn2, (3, 2); sign_function=global_sign)
    # T_reduced1[dum3, dum4, l1', u1, d1', r2, u2, d2'] = H_bond[dum3, dum4, dum5, dum6] * Tdn[dum5, l1', u1, d1', dum6, r2, u2, d2']
    T_reduced1 = contract(H_bond, Tdn, ((3, 4), (1, 5)); sign_function=global_sign)
    # T_up[dum3, l1, u1', d1, dum4, r2', u2', d2] = Tup1[dum3, l1, dum1, u1', d1] * Tup2[dum4, dum1, r2', u2', d2]
    T_up = contract(Tup1, Tup2, (3, 2); sign_function=global_sign)
    # T_reduced2[l1, u1', d1, r2', u2', d2, l1', u1, d1', r2, u2, d2'] = T_up[dum3, l1, u1', d1, dum4, r2', u2', d2] * T_reduced1[dum3, dum4, l1', u1, d1', r2, u2, d2']
    # T_reduced2[(l1, l1'), (r2', r2), (u1', u1), (u2', u2), (d1, d1'), (d2, d2')] <-- T_reduced2[l1, u1', d1, r2', u2', d2, l1', u1, d1', r2, u2, d2']
    T_reduced2 = contract(T_up, T_reduced1, ((1, 5), (1, 2)); perm=(1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12), sign_function=global_sign)
    # T_reduced3[L1', (r2', r2), (u1', u1), (u2', u2), (d1, d1'), (d2, d2')] = T_reduced2[(l1, l1'), (r2', r2), (u1', u1), (u2', u2), (d1, d1'), (d2, d2')]
    T_reduced2 = add_parity_sign(T_reduced2, 1; sign_function=global_sign)
    T_reduced3 = fuse(T_reduced2, (1, 2); index_type_fused=:out)
    # T_reduced4[L1', R2, (u1', u1), (u2', u2), (d1, d1'), (d2, d2')] <-- T_reduced3[L1', (r2', r2), (u1', u1), (u2', u2), (d1, d1'), (d2, d2')]
    T_reduced3 = add_perm_sign(T_reduced3, (1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11); sign_function=global_sign)
    T_reduced4 = fuse(T_reduced3, (2, 3); index_type_fused=:in)
    # T_reduced5[L1', R2, U1, (u2', u2), (d1, d1'), (d2, d2')] <-- T_reduced4[L1', R2, (u1', u1), (u2', u2), (d1, d1'), (d2, d2')]
    T_reduced4 = add_perm_sign(T_reduced4, (1, 2, 4, 3, 5, 6, 7, 8, 9, 10); sign_function=global_sign)
    T_reduced5 = fuse(T_reduced4, (3, 4); index_type_fused=:in)
    # T_reduced6[L1', R2, U1, U2, (d1, d1'), (d2, d2')] <-- T_reduced5[L1', R2, U1, (u2', u2), (d1, d1'), (d2, d2')]
    T_reduced5 = add_perm_sign(T_reduced5, (1, 2, 3, 5, 4, 6, 7, 8, 9); sign_function=global_sign)
    T_reduced6 = fuse(T_reduced5, (4, 5); index_type_fused=:in)
    # T_reduced7[L1', R2, U1, U2, D1', (d2, d2')] <-- T_reduced6[L1', R2, U1, U2, (d1, d1'), (d2, d2')]
    T_reduced6 = add_parity_sign(T_reduced6, 5; sign_function=global_sign)
    T_reduced7 = fuse(T_reduced6, (5, 6); index_type_fused=:out)
    # T_reduced8[L1', R2, U1, U2, D1', D2'] <-- T_reduced7[L1', R2, U1, U2, D1', (d2, d2')]
    T_reduced7 = add_parity_sign(T_reduced7, 6; sign_function=global_sign)
    T_reduced8 = fuse(T_reduced7, (6, 7); index_type_fused=:out)
end

""" 
Impurity bond along the x direction :
 
                                              [u1']
                                              ↗
                                            ↗
                                          ↗
                          [l1] ⟶⟶⟶ Tup[x, y] ⟶⟶⟶ [r1']  
                                      ↗ ↑
                                    ↗   ↑                                                                 [U]  
                                  ↗   [dum4]  #                                                         ↙
                              [dum1]    ↑   #                                                         ↙   
                              ↗         ↑ #                                                        ↙
                            ↗           #                                                        #
                          ↗           # ↑                                                      #
                        ↗           # [dum6]     [u1]                        [L1'] ⟵⟵⟵  #  ⟵⟵⟵ [R1]
        [l2] ⟶⟶⟶ Tup[x+1, y] ⟶⟶⟶⟶↑⟶ [r2']↙                ====>                    #   
                    ↗    ↑       #      ↑     ↙                                         #    
                  ↗    [dum3]  #        ↑   ↙                          [L2'] ⟵⟵⟵ # ⟵⟵⟵ [R2]    
                ↗[l1']⟵↑⟵⟵⟵⟵ Tdn[x, y] ⟵⟵⟵ [r1]                        #  
             [d2]        ↑   #          ↙                                        ↙    
                         ↑ #          ↙                                        ↙     
                         #          ↙                                        ↙ 
                       # ↑     [dum2]                                    [D']
                     #   ↑      ↙
                       [dum5] ↙
                         ↑  ↙
                         ↑↙
      [l2'] ⟵⟵⟵ Tdn[x+1, y] ⟵⟵⟵ [r2]
                      ↙
                    ↙
                  ↙
              [d2'] 
"""  

function reduced_tensor_x_bond(Tdn1::Grassmann{Q, 5}, Tdn2::Grassmann{Q, 5}, H_bond::Grassmann{Q, 4}) where {Q}

    Tup1 = conj(Tdn1; sign_function=global_sign)
    Tup2 = conj(Tdn2; sign_function=global_sign)

    # Tdn[dum6, l1', r1, u1, dum5, l2', r2, d2'] = Tdn1[dum6, l1', r1, u1, dum2] * Tdn2[dum5, l2', r2, dum2, d2']
    Tdn = contract(Tdn1, Tdn2, (5, 4); sign_function=global_sign) 
    # T_reduced1[dum3, dum4, l1', r1, u1, l2', r2, d2'] = H_bond[dum3, dum4, dum5, dum6] * Tdn[dum6, l1', r1, u1, dum5, l2', r2, d2']
    T_reduced1 = contract(H_bond, Tdn, ((3, 4), (5, 1)); sign_function=global_sign)
    # T_up[dum4, l1, r1', u1', dum3, l2, r2', d2] = Tup1[dum4, l1, r1', u1', dum1] * Tup2[dum3, l2, r2', dum1, d2]
    T_up = contract(Tup1, Tup2, (5, 4); sign_function=global_sign)
    # T_reduced2[l1, r1', u1', l2, r2', d2, l1', r1, u1, l2', r2, d2'] = T_up[dum4, l1, r1', u1', dum3, l2, r2', d2] * T_reduced1[dum3, dum4, l1', r1, u1, l2', r2, d2']
    # T_reduced2[(l1, l1'), (l2, l2'), (r1', r1), (r2', r2), (u1', u1), (d2, d2')] <-- T_reduced2[l1, r1', u1', l2, r2', d2, l1', r1, u1, l2', r2, d2']
    T_reduced2 = contract(T_up, T_reduced1, ((1, 5), (2, 1)); perm=(1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12), sign_function=global_sign)
    # T_reduced3[L1', (l2, l2'), (r1', r1), (r2', r2), (u1', u1), (d2, d2')] <-- T_reduced2[(l1, l1'), (l2, l2'),( r1', r1), (r2', r2), (u1', u1), (d2, d2')]
    T_reduced2 = add_parity_sign(T_reduced2, 1; sign_function=global_sign)
    T_reduced3 = fuse(T_reduced2, (1, 2); index_type_fused=:out)
    # T_reduced4[L1', L2', (r1', r1), (r2', r2), (u1', u1), (d2, d2')] <-- T_reduced3[L1', (l2, l2'), (r1', r1), (r2', r2), (u1', u1), (d2, d2')]
    T_reduced3 = add_parity_sign(T_reduced3, 2; sign_function=global_sign)
    T_reduced4 = fuse(T_reduced3, (2, 3); index_type_fused=:out)
    # T_reduced5[L1', L2', R1, (r2', r2), (u1', u1), (d2, d2')] <-- T_reduced4[L1', L2', (r1', r1), (r2', r2), (u1', u1), (d2, d2')]
    T_reduced4 = add_perm_sign(T_reduced4, (1, 2, 4, 3, 5, 6, 7, 8, 9, 10); sign_function=global_sign)
    T_reduced5 = fuse(T_reduced4, (3, 4); index_type_fused=:in)
    # T_reduced6[L1', L2', R1, R2, (u1', u1), (d2, d2')] <-- T_reduced5[L1', L2', R1, (r2', r2), (u1', u1), (d2, d2')]
    T_reduced5 = add_perm_sign(T_reduced5, (1, 2, 3, 5, 4, 6, 7, 8, 9); sign_function=global_sign)
    T_reduced6 = fuse(T_reduced5, (4, 5); index_type_fused=:in)
    # T_reduced7[L1', L2', R1, R2, U1, (d2, d2')] <-- T_reduced6[L1', L2', R1, R2, (u1', u1), (d2, d2')]
    T_reduced6 = add_perm_sign(T_reduced6, (1, 2, 3, 4, 6, 5, 7, 8); sign_function=global_sign)
    T_reduced7 = fuse(T_reduced6, (5, 6); index_type_fused=:in)
    # T_reduced8[L1', L2', R1, R2, U1, D2'] <-- T_reduced7[L1', L2', R1, R2, U1, (d2, d2')]
    T_reduced7 = add_parity_sign(T_reduced7, 6; sign_function=global_sign)
    T_reduced8 = fuse(T_reduced7, (6, 7); index_type_fused=:out)
end

function reduced_tensor(peps::Square_GPEPS{Q}, H_bond::Grassmann{Q, 4}) where {Q}

    Lx, Ly = size(peps)
  
    T_x_bond_mat = Matrix{Grassmann{Q, 6}}(undef, Lx, Ly)
    T_y_bond_mat = Matrix{Grassmann{Q, 6}}(undef, Lx, Ly)

    for r in 1:Lx, c in 1:Ly
        c_p1 = Nmod(c+1, Ly)
        r_p1 = Nmod(r+1, Lx)
        T_x_bond_mat[r, c] = reduced_tensor_x_bond(peps.A[r, c], peps.A[r_p1, c], H_bond)
        T_y_bond_mat[r, c] = reduced_tensor_y_bond(peps.A[r, c], peps.A[r, c_p1], H_bond)
    end

    return T_x_bond_mat, T_y_bond_mat
end


