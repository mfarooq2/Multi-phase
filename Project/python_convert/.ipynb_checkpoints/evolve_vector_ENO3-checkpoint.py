def evolve_vector_ENO3(phi, dx, dy, u_ext, v_ext):
    """
    Finds the amount of evolution under a vector field
    based force and using 3rd order accurate ENO scheme
    """
    delta = zeros(size(phi)+6);
    data_ext = zeros(size(phi)+6);
    data_ext[4:-3,4:-3] = phi;
    # first scan the rows
    for i in range(size(phi,1)):
        delta[i+3,:] = delta[i+3,:] + upwind_ENO3(data_ext[i+3,:], u_ext, dx);
    # then scan the columns
    for j in range(size(phi,2)):
        delta[:,j+3] = delta[:,j+3] + upwind_ENO3(data_ext[:,j+3], v_ext, dy);
    delta = delta[4:-3,4:-3];
    return delta