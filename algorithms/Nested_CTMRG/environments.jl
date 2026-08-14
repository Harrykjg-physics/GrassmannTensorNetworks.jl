
initialize_nested_environment(
    nested::NestedNetwork, 
    chi::Int, 
    chi_even::Int=div(chi, 2)) = CTMRGEnv(nested.network, chi, chi_even)
