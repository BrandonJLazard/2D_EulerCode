import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import fsolve



def differential_grid(x_grid, y_grid, epsilon):
    """
    Differential_grid solves the elliptical differential grid coupled partial differential equations where P,Q = 0

    Inputs:
    x_grid: 2D array of length JL x IL
    y_grid: 2D array of length JL x IL
    epsilon: the difference desired between the nth and nth+1 value for interior grid points

    Outputs:
    x_grid_new: 2D array of Length JL x IL with the new x positions of the grid points
    y_grid_new: 2D array of length JL x IL with the new y positions of the grid points
    n: number of iterations it took to converge 
    """

    #Assign IL and JL
    IL = len(x_grid[0])
    JL = len(x_grid[:, 0])


    #iteration count
    n = 0

    #Create error arrays and assign interior values to any value greater than 0
    x_err = np.zeros((JL, IL))
    y_err = np.zeros((JL, IL))
    x_err[1:JL-1, 1:IL-1] = 1
    y_err[1:JL-1, 1:IL-1] = 1



    x_grid_current = np.copy(x_grid)
    y_grid_current = np.copy(y_grid)

    x_grid_new = np.copy(x_grid)
    y_grid_new = np.copy(y_grid)

    #While the error in x or y is greater than epsilon, keep running the loop
    while (np.max(x_err) > epsilon) or (np.max(y_err) > epsilon):
        
        #Loop through every interior grid point and solve the differntial grid point equation in x and y
        for i in range(1, IL-1):
            for j in range(1, JL-1):

                #Calculate constants that depend on both x and y
                alpha = (((x_grid_current[j+1][i] - x_grid_current[j-1][i])/2)**(2)) + (((y_grid_current[j+1][i] - y_grid_current[j-1][i])/2)**(2))
                gamma = (((x_grid_current[j][i+1] - x_grid_current[j][i-1])/2)**(2)) + (((y_grid_current[j][i+1] - y_grid_current[j][i-1])/2)**(2))
                beta = ((x_grid_current[j][i+1] - x_grid_current[j][i-1])/2)*((x_grid_current[j+1][i] - x_grid_current[j-1][i])/2) + ((y_grid_current[j][i+1] - y_grid_current[j][i-1])/2)*((y_grid_current[j+1][i] - y_grid_current[j-1][i])/2)

                #Calculate the cross term that multiplies by 2beta
                x_cross = (x_grid_current[j+1][i+1] - x_grid_current[j+1][i-1] - x_grid_current[j-1][i+1] + x_grid_current[j-1][i-1])/4
                y_cross = (y_grid_current[j+1][i+1] - y_grid_current[j+1][i-1] - y_grid_current[j-1][i+1] + y_grid_current[j-1][i-1])/4

                #Calculate the term that multiplies by alpha
                x_alpha_term = x_grid_current[j][i+1] + x_grid_current[j][i-1]
                y_alpha_term = y_grid_current[j][i+1] + y_grid_current[j][i-1]

                #Calculate the term that multiplies by gamma
                x_gamma_term = x_grid_current[j+1][i] + x_grid_current[j-1][i]
                y_gamma_term = y_grid_current[j+1][i] + y_grid_current[j-1][i]

                #Calulate the value of x_ij at the n+1 iteration
                x_grid_new[j][i] = (-1/(2*alpha + 2*gamma))*((2*beta*x_cross) - (alpha*x_alpha_term) - (gamma*x_gamma_term))
                y_grid_new[j][i] = (-1/(2*alpha + 2*gamma))*((2*beta*y_cross) - (alpha*y_alpha_term) - (gamma*y_gamma_term))

        #Calculate error for interior points
        x_err = x_grid_new - x_grid_current
        y_err = y_grid_new - y_grid_current
        
        #After looping through the entire grid, we go up by n+1 and so the current grid is now the old new grid
        x_grid_current = np.copy(x_grid_new)
        y_grid_current = np.copy(y_grid_new)

        #Some diagnostics to see what the value of the error is if you so choose to print
        #print(np.linalg.norm((x_err)))
        #print(np.linalg.norm((y_err)))
        #print(np.max(y_err))
        #print(np.linalg.norm((y_err)))


        n += 1
    return (x_grid_new, y_grid_new, n)


def compute_geometric(i, j, x, y):
    '''
    compute_geometric solves the geometric quantities associated with a grid cell in the finite volume framework
    Note: This computes the quantities based on a counterclockwise notation with the i-1/2 normal vector pointing to the right and the j+1/2 vector pointing upwards

    Inputs:
    i, j: the ith column and the jth row of thr grid
    x: 2D grid of x positions
    y: 2D grid of y positions

    Outputs:
    nx: 1D array of the normal vectors in the x direction of every face
    ny: 1D array of the normal vectors in the y direction of every face
    S: The surface area of each face 
    V: Volume of the grid cell
    '''

    #Calculate the distance between the i+1/2 and i-1/2 face in both the x and y
    dy_iplus = y[j+1, i+1] - y[j, i+1]
    dy_imin = y[j+1, i] - y[j, i]
    dx_iplus = x[j+1, i+1] - x[j, i+1]
    dx_imin = x[j+1, i] - x[j, i]

    #Calculate the distance between the j+1/2 and j-1/2 face in both the x and y
    dy_jplus = y[j+1, i] - y[j+1, i+1] 
    dy_jmin = y[j, i] - y[j, i+1] 
    dx_jplus =  x[j+1, i] - x[j+1, i+1] 
    dx_jmin = x[j, i] - x[j, i+1] 

    #Calculate the surface area (in 2D its just a side length) of each face of the grid cell
    S_iplus = np.sqrt((dy_iplus**2) + (dx_iplus**2))
    S_imin = np.sqrt((dy_imin**2) + (dx_imin**2))
    S_jplus = np.sqrt((dy_jplus**2) + (dx_jplus**2))
    S_jmin = np.sqrt((dy_jmin**2) + (dx_jmin**2))

    #Calculate the normal directions of each face in both x and y
    nx_iplus = dy_iplus / S_iplus
    nx_imin = dy_imin / S_imin
    ny_iplus = -(dx_iplus / S_iplus)
    ny_imin = -(dx_imin / S_imin)

    nx_jplus = dy_jplus / S_jplus
    nx_jmin = dy_jmin / S_jmin
    ny_jplus = -(dx_jplus / S_jplus)
    ny_jmin = -(dx_jmin / S_jmin)

    #Calculate the volume of the grid cell
    S1 = (1/2)*np.abs(((x[j+1, i+1] - x[j, i])*(y[j+1, i] - y[j, i])) - ((x[j+1, i] - x[j, i])*(y[j+1, i+1] - y[j, i])))
    S2 = (1/2)*np.abs(((x[j, i+1] - x[j, i])*(y[j+1, i+1] - y[j, i])) - ((x[j+1, i+1] - x[j, i])*(y[j, i+1] - y[j, i])))
    V = S1 + S2

    #Combine everything into a neat array and return the arrays and volume
    nx = [nx_iplus, nx_imin, nx_jplus, nx_jmin]
    ny = [ny_iplus, ny_imin, ny_jplus, ny_jmin] 
    S = [S_iplus, S_imin, S_jplus, S_jmin]
 
    return (V, nx, ny, S)

def update_solidwall(U, x, y):
    '''
    update_solidwall updates the ghost cells of the top and lower boundary of the domain enforcing no flow through for the velocity normal to the surface
    Note: This only considers a 0th order extrapoloation of the values for pressure and density in the ghost cells

    Inputs:
    U: 3D array of length 4 x JL x IL where k = 1,2,4 corresponds to the conservative quantities being solved for in our code (density, momentum density in x and y, and energy)
    x: 2D grid of x positions
    y: 2D grid of y positions

    Outputs: 
    U: The same input but with updated boundary conditions along U_i,0 (bottom wall) and U_i,JL-1 (top wall)
    '''

    #Assume adiabatic index is 1.4 (diatomic gas)
    JL = len(U[0, :, 0])
    g = 1.4

    #Loop through every point in the ith direction
    for i in range(1, len(U[0, 0, :]) - 1):

        ########## LOWER WALL BC ###########################
        
        #Calculate the difference in height and length between the ijth cell and i+1,jth cell
        dy = y[0, i+1] - y[0, i]
        dx = x[0, i+1] - x[0, i]
        
        #Calculate the angle corresponding to the 
        theta = np.arctan(dy/dx)

        #Extrapolate quantities along the boundary of the cell
        u_1 = U[1, 1, i] / U[0, 1, i]
        v_1 = U[2, 1, i] / U[0, 1, i]
        rho_1 = U[0, 1, i]
        p_1 = (g-1)*(U[-1, 1, i] - (0.5/U[0, 1, i])*(U[1, 1, i]**2 + U[2, 1, i]**2))

        #Calculate the quantities of the ghost cell
        u_0 = np.cos(2*theta)*u_1 + np.sin(2*theta)*v_1
        v_0 = np.sin(2*theta)*u_1 - np.cos(2*theta)*v_1
        rho_0 = rho_1
        p_0 = p_1
        e_0 = p_0/(g-1) + (rho_0/2)*(u_0**2 + v_0**2)

        #Calculate the conservative quantities of the ghost cells
        U[0, 0, i] = rho_0
        U[1, 0, i] = rho_0*u_0
        U[2, 0, i] = rho_0*v_0
        U[3, 0, i] = e_0


        ######## UPPER WALL BC ########################################

        #Calculate the difference in height and length between the ijth cell and i+1,jth cell
        dy = y[JL-1, i+1] - y[JL-1, i]
        dx = x[JL-1, i+1] - x[JL-1, i]

        #Calculate the angle corresponding to the 
        theta = np.arctan(dy/dx)

        #Extrapolate quantities along the boundary of the cell
        u_1 = U[1, JL-2, i] / U[0, JL-2, i]
        v_1 = U[2, JL-2, i] / U[0, JL-2, i]
        rho_1 = U[0, JL-2, i]
        p_1 = (g-1)*(U[-1, JL-2, i] - (0.5/U[0, JL-2, i])*(U[1, JL-2, i]**2 + U[2, JL-2, i]**2))

        #Calculate the quantities of the ghost cell
        u_0 = np.cos(2*theta)*u_1 + np.sin(2*theta)*v_1
        v_0 = np.sin(2*theta)*u_1 - np.cos(2*theta)*v_1
        rho_0 = rho_1
        p_0 = p_1
        e_0 = p_0/(g-1) + (rho_0/2)*(u_0**2 + v_0**2)

        #Calculate the conservative quantities of the ghost cells
        U[0, JL-1, i] = rho_0
        U[1, JL-1, i] = rho_0*u_0
        U[2, JL-1, i] = rho_0*v_0
        U[3, JL-1, i] = e_0

    return U


def super_inlet(U, M, rho, P):
    '''
    super_inlet updates the ghost cells of the leftmost inlet boundary based on initial flow conditions specified by the user 
    Note: g, R assume that incoming flow is air

    Inputs:
    U: 3D array of length 4 x JL x IL where k = 1,2,4 corresponds to the conservative quantities being solved for in our code (density, momentum density in x and y, and energy)
    M: Value of the incoming Mach number
    rho: Density of the incoming flow
    P: Pressure of the incoming flow

    Outputs: 
    U: The same input but with updated boundary conditions along U_0,j (leftmost inlet)
    '''

    #Assume incoming flow is air
    g = 1.4
    R = 287    

    #Calculate conservative quantities
    T = P/(rho*R)
    c = np.sqrt(R*T*g)
    u = M*c
    v = 0

    e = P/(g-1) + (rho/2)*(u**2 + v**2)

    #Loop through every row on the left boudnary 
    for j in range(len(U[0, :, 0])):
        U[0, j, 0] =  rho
        U[1, j, 0] =  rho*u
        U[2, j, 0] =  rho*v
        U[3, j, 0] =  e
        
    return U

def super_outlet(U):
    '''
    super_oulet updates the ghost cells of the rightmost outlet boundary based on a 0th order extrapolation of the neighboring cell  

    Inputs:
    U: 3D array of length 4 x JL x IL where k = 1,2,4 corresponds to the conservative quantities being solved for in our code (density, momentum density in x and y, and energy)

    Outputs: 
    U: The same input but with updated boundary conditions along U_IL-1,j (rightmost inlet)
    '''

    U[:, :, -1] = U[:, :, -2]
    return U


def sub_outlet(U, dt, dx):
    IL = len(U[0, 0, :])
    P = 10**(5)
    g = 1.4
    R = 287
    rho = U[0]
    p = (g-1)*(U[-1] - (0.5/U[0])*(U[1]**2 + U[2]**2))
    T = p / (R*rho)
    c = np.sqrt(R*g*T)


    u = U[1] / U[0]
    v = U[2] / U[0]
    rho_new = np.copy(rho[:, IL-1])
    u_new = np.copy(u[:, IL-1])
    #print(p[:, IL-2])
    rho_new = rho[:, IL-1] - (u[:, IL-1]*(dt/dx)*((rho[:, IL-1] - rho[:, IL-2]) - ((1/(c[:, IL-1]**2))*(p[:, IL-1] - p[:, IL-2]))))
    #print(rho_new)
    u_new = u[:, IL-1] - (1/(rho[:, IL-1]*c[:, IL-1]))*(u[:, IL-1]+c[:, IL-1])*(dt/dx)*((p[:, IL-1] - p[:, IL-2]) + ((rho[:, IL-1]*c[:, IL-1])*(u[:, IL-1] - u[:, IL-2])))
    e_new = P/(g-1) + (rho_new/2)*(u_new**2 + v[:, IL-1]**2)


    U[0, :, IL-1] = rho_new
    U[1, :, IL-1] = rho_new*u_new
    U[2, :, IL-1] = rho_new*v[:, IL-1]
    U[3, :, IL-1] = e_new

    return U


def sub_inlet(U, Tt, Pt, dt, dx):
    IL = len(U[0, 0, :])
    g = 1.4
    R = 287
    rat = (g-1)/(g+1)
    a_star = np.sqrt((2*g*R*Tt)/(g+1))

    rho = U[0]
    p = (g-1)*(U[-1] - (0.5/U[0])*(U[1]**2 + U[2]**2))
    T = p / (R*rho)
    c = np.sqrt(R*g*T)
    u = U[1] / U[0]
    v = U[2] / U[0]

    print(u)

    u_new = np.copy(u[:, 0])
    term1 = ((((u[:, 0])*(-2*Pt*g))/((a_star**2)*(g+1))))
    print('velcity' + str(u[:, 0]))

    r = ((((u[:, 0])*(-2*Pt*g))/((a_star**2)*(g+1)))*((1 - rat*((u[:, 0]**2) / (a_star**2)))**((g/(g-1)) - 1))) - (rho[:, 0]*c[:, 0])

    u_new = u[:, 0] - (1/r)*(u[:, 0] - c[:, 0])*(dt/dx)*((p[:, 1] - p[:, 0]) - ((rho[:, 0]*c[:, 0])*(u[:, 1] - u[:, 0])))

    p_new = Pt*((1 - (rat)*((u_new**2)/(a_star**2)))**(g / (g-1)))

    T_new = Tt*((1 - (rat)*((u_new**2)/(a_star**2))))
    rho_new = p_new / (R*T_new)
    print('rho_new = ' + str(rho_new))

    e_new = p_new/(g-1) + (rho_new/2)*(u_new**2 + v[:, 0]**2)
    

    U[0, :, 0] = rho_new
    U[1, :, 0] = rho_new*u_new
    U[2, :, 0] = rho_new*v[:, 0]
    U[3, :, 0] = e_new

    return U


def compute_interior_LF(U, i, j, x, y, dt, alpha = 1.5):
    '''
    compute_interior_LF computes the interior points of the conservative variables of the Euler equation in 2D.
    This function employs the Lax-Friedrich method which is numerical disspative. This scheme is 1st order in time and 1st order in space. 
    Note: g, R assume that incoming flow is air

    Inputs:
    U: 3D array of length 4 x JL x IL where k = 1,2,4 corresponds to the conservative quantities being solved for in our code (density, momentum density in x and y, and energy)
    i, j: the ith column and the jth row of thr grid
    x: 2D grid of x positions
    y: 2D grid of y positions
    dt: The timestep corresponding to the iteration of the entire grid
    alpha: Disspiation coefficient for the LF scheme. In general alpha>1.0 in for the LF scheme to be stable

    Outputs: 
    U_new: a 1D array of the solved conservative variables for the ij grid cell
    '''

    #Create empty array to replace new values with
    U_new = np.zeros_like(U[:, 0, 0])

    #Assume interior is Air
    g = 1.4
    R = 287

    #Calculate the lenght in x and y
    JL = len(U[0, :, 0])
    IL = len(U[0, 0, :])

    #Calculate flow quantities of the entire grid
    u = U[1] / U[0]
    v = U[2] / U[0]
    rho = U[0]
    p = (g-1)*(U[-1] - (0.5/U[0])*(U[1]**2 + U[2]**2))
    T = p / (R*rho)
    c = np.sqrt(R*g*T)

    #Average the velocities and sound speed to make an approximate of the value at each face of the grid cell
    u_iplus = (u[j, i] + u[j, i+1]) / 2
    u_imin = (u[j, i] + u[j, i-1]) / 2
    u_jplus = (u[j, i] + u[j+1, i]) / 2
    u_jmin = (u[j, i] + u[j-1, i]) / 2

    v_iplus = (v[j, i] + v[j, i+1]) / 2
    v_imin = (v[j, i] + v[j, i-1]) / 2
    v_jplus = (v[j, i] + v[j+1, i]) / 2
    v_jmin = (v[j, i] + v[j-1, i]) / 2

    c_iplus = (c[j, i] + c[j, i+1]) / 2
    c_imin = (c[j, i] + c[j, i-1]) / 2
    c_jplus = (c[j, i] + c[j+1, i]) / 2
    c_jmin = (c[j, i] + c[j-1, i]) / 2

    #Compute the geometric quantities of the ijth grid cell
    V, nx, ny, S = compute_geometric(i, j, x, y)

    #Calculate the vector velocities through each face
    u_prime_iplus = u_iplus*nx[0] + v_iplus*ny[0]
    u_prime_imin = u_imin*nx[1] + v_imin*ny[1]
    u_prime_jplus = u_jplus*nx[2] + v_jplus*ny[2]
    u_prime_jmin = u_jmin*nx[3] + v_jmin*ny[3] 

    #Calculate the damping coefficient of each face
    l_iplus = alpha*(np.abs(u_prime_iplus) + c_iplus)
    l_imin = alpha*(np.abs(u_prime_imin) + c_imin)
    l_jplus = alpha*(np.abs(u_prime_jplus) + c_jplus)
    l_jmin = alpha*(np.abs(u_prime_jmin) + c_jmin)

    #Create values of the fluxes through each face
    F_ij_iplus = np.zeros(4)
    F_ij_imin = np.zeros(4)
    F_ij_jplus = np.zeros(4)
    F_ij_jmin = np.zeros(4)
    F_iplusj = np.zeros(4)
    F_iminj = np.zeros(4)
    F_ijplus = np.zeros(4)
    F_ijmin = np.zeros(4)

    ################ CALCULATES FLUXES THROUGH EACH FACE ######################

    #center ij for iplus face
    P = (g-1)*(U[-1, j, i] - (0.5/U[0, j, i])*(U[1, j, i]**2 + U[2, j, i]**2))
    F_ij_iplus[0] =  U[0, j, i]*u_prime_iplus
    F_ij_iplus[1] = U[0, j, i]*(U[1, j, i]/U[0, j, i])*u_prime_iplus + P*nx[0]
    F_ij_iplus[2] = U[0, j, i]*(U[2, j, i]/U[0, j, i])*u_prime_iplus + P*ny[0]
    F_ij_iplus[3] = (U[3, j, i]+P)*u_prime_iplus

    #center ij for iminus face
    P = (g-1)*(U[-1, j, i] - (0.5/U[0, j, i])*(U[1, j, i]**2 + U[2, j, i]**2))
    F_ij_imin[0] =  U[0, j, i]*u_prime_imin
    F_ij_imin[1] = U[0, j, i]*(U[1, j, i]/U[0, j, i])*u_prime_imin + P*nx[1]
    F_ij_imin[2] = U[0, j, i]*(U[2, j, i]/U[0, j, i])*u_prime_imin + P*ny[1]
    F_ij_imin[3] = (U[3, j, i]+P)*u_prime_imin

    #center ij for jplus face
    P = (g-1)*(U[-1, j, i] - (0.5/U[0, j, i])*(U[1, j, i]**2 + U[2, j, i]**2))
    F_ij_jplus[0] =  U[0, j, i]*u_prime_jplus
    F_ij_jplus[1] = U[0, j, i]*(U[1, j, i]/U[0, j, i])*u_prime_jplus + P*nx[2]
    F_ij_jplus[2] = U[0, j, i]*(U[2, j, i]/U[0, j, i])*u_prime_jplus + P*ny[2]
    F_ij_jplus[3] = (U[3, j, i]+P)*u_prime_jplus

    #center ij for jmin face
    P = (g-1)*(U[-1, j, i] - (0.5/U[0, j, i])*(U[1, j, i]**2 + U[2, j, i]**2))
    F_ij_jmin[0] =  U[0, j, i]*u_prime_jmin
    F_ij_jmin[1] = U[0, j, i]*(U[1, j, i]/U[0, j, i])*u_prime_jmin + P*nx[3]
    F_ij_jmin[2] = U[0, j, i]*(U[2, j, i]/U[0, j, i])*u_prime_jmin + P*ny[3]
    F_ij_jmin[3] = (U[3, j, i]+P)*u_prime_jmin

    #i+1 cell for iplus face
    P = (g-1)*(U[-1, j, i+1] - (0.5/U[0, j, i+1])*(U[1, j, i+1]**2 + U[2, j, i+1]**2))
    F_iplusj[0] =  U[0, j, i+1]*u_prime_iplus
    F_iplusj[1] = U[0, j, i+1]*(U[1, j, i+1]/U[0, j, i+1])*u_prime_iplus + P*nx[0]
    F_iplusj[2] = U[0, j, i+1]*(U[2, j, i+1]/U[0, j, i+1])*u_prime_iplus + P*ny[0]
    F_iplusj[3] = (U[3, j, i+1]+P)*u_prime_iplus

    #i-1 cell for imin face
    P = (g-1)*(U[-1, j, i-1] - (0.5/U[0, j, i-1])*(U[1, j, i-1]**2 + U[2, j, i-1]**2))
    F_iminj[0] =  U[0, j, i-1]*u_prime_imin
    F_iminj[1] = U[0, j, i-1]*(U[1, j, i-1]/U[0, j, i-1])*u_prime_imin + P*nx[1]
    F_iminj[2] = U[0, j, i-1]*(U[2, j, i-1]/U[0, j, i-1])*u_prime_imin + P*ny[1]
    F_iminj[3] = (U[3, j, i-1]+P)*u_prime_imin

    #j+1 cell for jplus face
    P = (g-1)*(U[-1, j+1, i] - (0.5/U[0, j+1, i])*(U[1, j+1, i]**2 + U[2, j+1, i]**2))
    F_ijplus[0] =  U[0, j+1, i]*u_prime_jplus
    F_ijplus[1] = U[0, j+1, i]*(U[1, j+1, i]/U[0, j+1, i])*u_prime_jplus + P*nx[2]
    F_ijplus[2] = U[0, j+1, i]*(U[2, j+1, i]/U[0, j+1, i])*u_prime_jplus + P*ny[2]
    F_ijplus[3] = (U[3, j+1, i]+P)*u_prime_jplus
    
    #j-1 cell for jmin face
    P = (g-1)*(U[-1, j-1, i] - (0.5/U[0, j-1, i])*(U[1, j-1, i]**2 + U[2, j-1, i]**2))
    F_ijmin[0] =  U[0, j-1, i]*u_prime_jmin
    F_ijmin[1] = U[0, j-1, i]*(U[1, j-1, i]/U[0, j-1, i])*u_prime_jmin + P*nx[3]
    F_ijmin[2] = U[0, j-1, i]*(U[2, j-1, i]/U[0, j-1, i])*u_prime_jmin + P*ny[3]
    F_ijmin[3] = (U[3, j-1, i]+P)*u_prime_jmin


    #Calculate the fluxes through each surface
    F_LF_iplus = (0.5)*(F_ij_iplus + F_iplusj) - (0.5*l_iplus)*(U[:, j, i+1] - U[:, j, i])
    F_LF_imin = (0.5)*(F_iminj + F_ij_imin) - (0.5*l_imin)*(U[:, j, i] - U[:, j, i-1])
    F_LF_jplus = (0.5)*(F_ij_jplus + F_ijplus) - (0.5*l_jplus)*(U[:, j+1, i] - U[:, j, i])
    F_LF_jmin = (0.5)*(F_ijmin + F_ij_jmin) - (0.5*l_jmin)*(U[:, j, i] - U[:, j-1, i])   

    
    #Compute the conservative quantity of each grid cell
    U_new = U[:, j, i] - ((dt/V)*(F_LF_iplus*S[0] - F_LF_imin*S[1] + F_LF_jplus*S[2] - F_LF_jmin*S[3]))

    return U_new

def compute_interior_SW(U, i, j, x, y, dt):
    '''
    compute_interior_SW computes the interior points of the conservative variables of the Euler equation in 2D.
    This function employs the Steger-Warming Flux Splitting method which is numerical disspative. This scheme is 1st order in time and 1st order in space. 
    Note: g, R assume that incoming flow is air

    Inputs:
    U: 3D array of length 4 x JL x IL where k = 1,2,4 corresponds to the conservative quantities being solved for in our code (density, momentum density in x and y, and energy)
    i, j: the ith column and the jth row of thr grid
    x: 2D grid of x positions
    y: 2D grid of y positions
    dt: The timestep corresponding to the iteration of the entire grid
    alpha: Disspiation coefficient for the LF scheme. In general alpha>1.0 in for the LF scheme to be stable

    Outputs: 
    U_new: a 1D array of the solved conservative variables for the ij grid cell
    '''

    #Create empty array to replace new values with
    U_new = np.zeros_like(U[:, 0, 0])

    #Calculate geoemtric quantities for the grid cell
    V, nx, ny, S = compute_geometric(i, j, x, y)

    #Assume interior is Air
    g = 1.4
    R = 287

    #Calculate the lenght in x and y
    JL = len(U[0, :, 0])
    IL = len(U[0, 0, :])
    

    #Calculate flow quantities of the entire grid
    u = U[1] / U[0]
    v = U[2] / U[0]
    rho = U[0]
    p = (g-1)*(U[-1] - (0.5/U[0])*(U[1]**2 + U[2]**2))
    T = p / (R*rho)
    c = np.sqrt(R*g*T)

    #Average the velocities and sound speed to make an approximate of the value at each face of the grid cell
    u_iplus = (u[j, i] + u[j, i+1]) / 2
    u_imin = (u[j, i] + u[j, i-1]) / 2
    u_jplus = (u[j, i] + u[j+1, i]) / 2
    u_jmin = (u[j, i] + u[j-1, i]) / 2

    v_iplus = (v[j, i] + v[j, i+1]) / 2
    v_imin = (v[j, i] + v[j, i-1]) / 2
    v_jplus = (v[j, i] + v[j+1, i]) / 2
    v_jmin = (v[j, i] + v[j-1, i]) / 2

    c_iplus = (c[j, i] + c[j, i+1]) / 2
    c_imin = (c[j, i] + c[j, i-1]) / 2
    c_jplus = (c[j, i] + c[j+1, i]) / 2
    c_jmin = (c[j, i] + c[j-1, i]) / 2


    #Calculate the vector velocities through each face
    u_prime_iplus = u_iplus*nx[0] + v_iplus*ny[0]
    u_prime_imin = u_imin*nx[1] + v_imin*ny[1]
    u_prime_jplus = u_jplus*nx[2] + v_jplus*ny[2]
    u_prime_jmin = u_jmin*nx[3] + v_jmin*ny[3] 

    v_prime_iplus = -1*u_iplus*ny[0] + v_iplus*nx[0]
    v_prime_imin = -1*u_imin*ny[1] + v_imin*nx[1]
    v_prime_jplus = -1*u_jplus*ny[2] + v_jplus*nx[2]
    v_prime_jmin = -1*u_jmin*ny[3] + v_jmin*nx[3] 

    B = g - 1
    a = ((u**2) + (v**2)) /2

    #L^+_ij for iplus face
    L_iplus_ij = np.array([[(u_prime_iplus + np.abs(u_prime_iplus)) / 2, 0, 0, 0],
     [0, ((u_prime_iplus + c[j, i]) + np.abs((u_prime_iplus + c[j, i]))) / 2, 0, 0],
     [0, 0, (u_prime_iplus + np.abs(u_prime_iplus)) / 2, 0],
     [0, 0, 0, ((u_prime_iplus - c[j, i]) + np.abs((u_prime_iplus - c[j, i]))) / 2]])
    
    P_iplus_ij = np.array([[1, 1/(2*(c[j, i]**2)), 0, 1/(2*(c[j, i]**2))], 
     [u[j, i], u[j, i]/(2*(c[j, i]**2)) + nx[0]/(2*c[j, i]), -1*ny[0]*rho[j, i], u[j, i]/(2*(c[j, i]**2)) - nx[0]/(2*c[j, i])],
     [v[j, i], v[j, i]/(2*(c[j, i]**2)) + ny[0]/(2*c[j, i]), nx[0]*rho[j, i], v[j, i]/(2*(c[j, i]**2)) - ny[0]/(2*c[j, i])],
     [a[j, i], a[j, i]/(2*(c[j, i]**2)) + u_prime_iplus/(2*c[j, i]) + 1/(2*B), rho[j, i]*v_prime_iplus, a[j, i]/(2*(c[j, i]**2)) - u_prime_iplus/(2*c[j, i]) + 1/(2*B)]])
    

    #L^+_ij for jplus face
    L_jplus_ij = np.array([[(u_prime_jplus + np.abs(u_prime_jplus)) / 2, 0, 0, 0],
     [0, ((u_prime_jplus + c[j, i]) + np.abs((u_prime_jplus + c[j, i]))) / 2, 0, 0],
     [0, 0, (u_prime_jplus + np.abs(u_prime_jplus)) / 2, 0],
     [0, 0, 0, ((u_prime_jplus - c[j, i]) + np.abs((u_prime_jplus - c[j, i]))) / 2]])
    
    P_jplus_ij = np.array([[1, 1/(2*(c[j, i]**2)), 0, 1/(2*(c[j, i]**2))], 
     [u[j, i], u[j, i]/(2*(c[j, i]**2)) + nx[2]/(2*c[j, i]), -1*ny[2]*rho[j, i], u[j, i]/(2*(c[j, i]**2)) - nx[2]/(2*c[j, i])],
     [v[j, i], v[j, i]/(2*(c[j, i]**2)) + ny[2]/(2*c[j, i]), nx[2]*rho[j, i], v[j, i]/(2*(c[j, i]**2)) - ny[2]/(2*c[j, i])],
     [a[j, i], a[j, i]/(2*(c[j, i]**2)) + u_prime_jplus/(2*c[j, i]) + 1/(2*B), rho[j, i]*v_prime_jplus, a[j, i]/(2*(c[j, i]**2)) - u_prime_jplus/(2*c[j, i]) + 1/(2*B)]])
    

    #L^-_ij for iminus face
    L_imin_ij = np.array([[(u_prime_imin - np.abs(u_prime_imin)) / 2, 0, 0, 0],
     [0, ((u_prime_imin + c[j, i]) - np.abs((u_prime_imin + c[j, i]))) / 2, 0, 0],
     [0, 0, (u_prime_imin - np.abs(u_prime_imin)) / 2, 0],
     [0, 0, 0, ((u_prime_imin - c[j, i]) - np.abs((u_prime_imin - c[j, i]))) / 2]])
    
    P_imin_ij = np.array([[1, 1/(2*(c[j, i]**2)), 0, 1/(2*(c[j, i]**2))], 
     [u[j, i], u[j, i]/(2*(c[j, i]**2)) + nx[1]/(2*c[j, i]), -1*ny[1]*rho[j, i], u[j, i]/(2*(c[j, i]**2)) - nx[1]/(2*c[j, i])],
     [v[j, i], v[j, i]/(2*(c[j, i]**2)) + ny[1]/(2*c[j, i]), nx[1]*rho[j, i], v[j, i]/(2*(c[j, i]**2)) - ny[1]/(2*c[j, i])],
     [a[j, i], a[j, i]/(2*(c[j, i]**2)) + u_prime_imin/(2*c[j, i]) + 1/(2*B), rho[j, i]*v_prime_imin, a[j, i]/(2*(c[j, i]**2)) - u_prime_imin/(2*c[j, i]) + 1/(2*B)]])
    

    #L^-_ij for jminus face
    L_jmin_ij = np.array([[(u_prime_jmin - np.abs(u_prime_jmin)) / 2, 0, 0, 0],
     [0, ((u_prime_jmin + c[j, i]) - np.abs((u_prime_jmin + c[j, i]))) / 2, 0, 0],
     [0, 0, (u_prime_jmin - np.abs(u_prime_jmin)) / 2, 0],
     [0, 0, 0, ((u_prime_jmin - c[j, i]) - np.abs((u_prime_jmin - c[j, i]))) / 2]])    
    
    P_jmin_ij = np.array([[1, 1/(2*(c[j, i]**2)), 0, 1/(2*(c[j, i]**2))], 
     [u[j, i], u[j, i]/(2*(c[j, i]**2)) + nx[-1]/(2*c[j, i]), -1*ny[-1]*rho[j, i], u[j, i]/(2*(c[j, i]**2)) - nx[-1]/(2*c[j, i])],
     [v[j, i], v[j, i]/(2*(c[j, i]**2)) + ny[-1]/(2*c[j, i]), nx[-1]*rho[j, i], v[j, i]/(2*(c[j, i]**2)) - ny[-1]/(2*c[j, i])],
     [a[j, i], a[j, i]/(2*(c[j, i]**2)) + u_prime_jmin/(2*c[j, i]) + 1/(2*B), rho[j, i]*v_prime_jmin, a[j, i]/(2*(c[j, i]**2)) - u_prime_jmin/(2*c[j, i]) + 1/(2*B)]])


    #L^-_i+1,j 
    L_iplusj = np.array([[(u_prime_iplus - np.abs(u_prime_iplus)) / 2, 0, 0, 0],
     [0, ((u_prime_iplus + c[j, i+1]) - np.abs((u_prime_iplus + c[j, i+1]))) / 2, 0, 0],
     [0, 0, (u_prime_iplus - np.abs(u_prime_iplus)) / 2, 0],
     [0, 0, 0, ((u_prime_iplus - c[j, i+1]) - np.abs((u_prime_iplus - c[j, i+1]))) / 2]])

    P_iplusj = np.array([[1, 1/(2*(c[j, i+1]**2)), 0, 1/(2*(c[j, i+1]**2))], 
     [u[j, i+1], u[j, i+1]/(2*(c[j, i+1]**2)) + nx[0]/(2*c[j, i+1]), -1*ny[0]*rho[j, i+1], u[j, i+1]/(2*(c[j, i+1]**2)) - nx[0]/(2*c[j, i+1])],
     [v[j, i+1], v[j, i+1]/(2*(c[j, i+1]**2)) + ny[0]/(2*c[j, i+1]), nx[0]*rho[j, i+1], v[j, i+1]/(2*(c[j, i+1]**2)) - ny[0]/(2*c[j, i+1])],
     [a[j, i+1], a[j, i+1]/(2*(c[j, i+1]**2)) + u_prime_iplus/(2*c[j, i+1]) + 1/(2*B), rho[j, i+1]*v_prime_iplus, a[j, i+1]/(2*(c[j, i+1]**2)) - u_prime_iplus/(2*c[j, i+1]) + 1/(2*B)]]) 


    #L^-_ij+1
    L_ijplus = np.array([[(u_prime_jplus - np.abs(u_prime_jplus)) / 2, 0, 0, 0],
     [0, ((u_prime_jplus + c[j+1, i]) - np.abs((u_prime_jplus + c[j+1, i]))) / 2, 0, 0],
     [0, 0, (u_prime_jplus - np.abs(u_prime_jplus)) / 2, 0],
     [0, 0, 0, ((u_prime_jplus - c[j+1, i]) - np.abs((u_prime_jplus - c[j+1, i]))) / 2]])
    
    P_ijplus = np.array([[1, 1/(2*(c[j+1, i]**2)), 0, 1/(2*(c[j+1, i]**2))], 
     [u[j+1, i], u[j+1, i]/(2*(c[j+1, i]**2)) + nx[2]/(2*c[j+1, i]), -1*ny[2]*rho[j+1, i], u[j+1, i]/(2*(c[j+1, i]**2)) - nx[2]/(2*c[j+1, i])],
     [v[j+1, i], v[j+1, i]/(2*(c[j+1, i]**2)) + ny[2]/(2*c[j+1, i]), nx[2]*rho[j+1, i], v[j+1, i]/(2*(c[j+1, i]**2)) - ny[2]/(2*c[j+1, i])],
     [a[j+1, i], a[j+1, i]/(2*(c[j+1, i]**2)) + u_prime_jplus/(2*c[j+1, i]) + 1/(2*B), rho[j+1, i]*v_prime_jplus, a[j+1, i]/(2*(c[j+1, i]**2)) - u_prime_jplus/(2*c[j+1, i]) + 1/(2*B)]])
    

    #L^+_i-1,j
    L_iminj = np.array([[(u_prime_imin + np.abs(u_prime_imin)) / 2, 0, 0, 0],
     [0, ((u_prime_imin + c[j, i-1]) + np.abs((u_prime_imin + c[j, i-1]))) / 2, 0, 0],
     [0, 0, (u_prime_imin + np.abs(u_prime_imin)) / 2, 0],
     [0, 0, 0, ((u_prime_imin - c[j, i-1]) + np.abs((u_prime_imin - c[j, i-1]))) / 2]])
    
    P_iminj = np.array([[1, 1/(2*(c[j, i-1]**2)), 0, 1/(2*(c[j, i-1]**2))], 
     [u[j, i-1], u[j, i-1]/(2*(c[j, i-1]**2)) + nx[1]/(2*c[j, i-1]), -1*ny[1]*rho[j, i-1], u[j, i-1]/(2*(c[j, i-1]**2)) - nx[1]/(2*c[j, i-1])],
     [v[j, i-1], v[j, i-1]/(2*(c[j, i-1]**2)) + ny[1]/(2*c[j, i-1]), nx[1]*rho[j, i-1], v[j, i-1]/(2*(c[j, i-1]**2)) - ny[1]/(2*c[j, i-1])],
     [a[j, i-1], a[j, i-1]/(2*(c[j, i-1]**2)) + u_prime_imin/(2*c[j, i-1]) + 1/(2*B), rho[j, i-1]*v_prime_imin, a[j, i-1]/(2*(c[j, i-1]**2)) - u_prime_imin/(2*c[j, i-1]) + 1/(2*B)]])


    #L^+_ij-1 
    L_ijmin = np.array([[(u_prime_jmin + np.abs(u_prime_jmin)) / 2, 0, 0, 0],
     [0, ((u_prime_jmin + c[j-1, i]) + np.abs((u_prime_jmin + c[j-1, i]))) / 2, 0, 0],
     [0, 0, (u_prime_jmin + np.abs(u_prime_jmin)) / 2, 0],
     [0, 0, 0, ((u_prime_jmin - c[j-1, i]) + np.abs((u_prime_jmin - c[j-1, i]))) / 2]])  
    
    P_ijmin = np.array([[1, 1/(2*(c[j-1, i]**2)), 0, 1/(2*(c[j-1, i]**2))], 
     [u[j-1, i], u[j-1, i]/(2*(c[j-1, i]**2)) + nx[-1]/(2*c[j-1, i]), -1*ny[-1]*rho[j-1, i], u[j-1, i]/(2*(c[j-1, i]**2)) - nx[-1]/(2*c[j-1, i])],
     [v[j-1, i], v[j-1, i]/(2*(c[j-1, i]**2)) + ny[-1]/(2*c[j-1, i]), nx[-1]*rho[j-1, i], v[j-1, i]/(2*(c[j-1, i]**2)) - ny[-1]/(2*c[j-1, i])],
     [a[j-1, i], a[j-1, i]/(2*(c[j-1, i]**2)) + u_prime_jmin/(2*c[j-1, i]) + 1/(2*B), rho[j-1, i]*v_prime_jmin, a[j-1, i]/(2*(c[j-1, i]**2)) - u_prime_jmin/(2*c[j-1, i]) + 1/(2*B)]])

    #Calculate all the fluxes
    F_iplusj = (P_iplus_ij @ L_iplus_ij @ np.linalg.inv(P_iplus_ij) @ (U[:, j, i])) + (P_iplusj @ L_iplusj @ np.linalg.inv(P_iplusj) @ (U[:, j, i+1]))
    F_ijplus = (P_jplus_ij @ L_jplus_ij @ np.linalg.inv(P_jplus_ij) @ (U[:, j, i])) + (P_ijplus @ L_ijplus @ np.linalg.inv(P_ijplus) @ (U[:, j+1, i]))
    F_iminj = (P_imin_ij @ L_imin_ij @ np.linalg.inv(P_imin_ij) @ (U[:, j, i])) + (P_iminj @ L_iminj @ np.linalg.inv(P_iminj) @ (U[:, j, i-1]))
    F_ijmin = (P_jmin_ij @ L_jmin_ij @ np.linalg.inv(P_jmin_ij) @ (U[:, j, i])) + (P_ijmin @ L_ijmin @ np.linalg.inv(P_ijmin) @ (U[:, j-1, i]))

    
    #Compute the conservative quantity of each grid cell
    U_new = U[:, j, i] - ((dt/V)*(F_iplusj*S[0] - F_iminj*S[1] + F_ijplus*S[2] - F_ijmin*S[3]))

    return U_new


def update_time(x, y, CFL, u, v, c):

    JL = len(x[:, 0])
    IL = len(x[0, :])
    dx = []
    dy = []
    for j in range(JL-1):
        for i in range(IL-1):
            dx.append(x[j, i+1] - x[j, i])
            dy.append(y[j+1, i] - y[j, i])
            dx.append(x[j+1, i] - x[j, i+1])
            dy.append(y[j+1, i+1] - y[j, i])
    
    dx = np.max(dx)
    dy = np.max(dy)
    u = np.max(u)
    v = np.max(v)
    c = np.max(c)

    dt = CFL / ((np.abs(u) / dx) + (np.abs(v) / dy) + (c*np.sqrt((dx**(-2) + (dy**(-2))))))

    return dt
