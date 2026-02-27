import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

L = 128
Dvs = [2.3, 3, 5, 9]
time_step = 0.01
n_steps = 30000
q = 1
Du = 1
a = 3
b = 8

u = 3
v = 8/ 3

def initiation(L, u, v):
    u0 = np.zeros((L, L))
    v0 = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            u0[i, j] = u * (1 + 0.1 * np.random.rand())
            v0[i, j] = v * (1 + 0.1 * np.random.rand())
    return u0, v0

def calculate_k_values(DA, J11, J22, DI, detJ):
    d = DI/DA
    divide = 1 / (2 * d * DA)
    sqrt_term = np.sqrt((J11 + J22)**2 - 4 * d * detJ)
    k_squared = divide * (d*J11 + J22 + sqrt_term)
    k_squared_2 = divide * (d*J11 + J22 - sqrt_term)
    return k_squared, k_squared_2

def laplacian(u):
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis = 0) + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u)

def solutions_rectangular_boundary(j, L, l, q, x, y):
    # x and y are the coordinates of the point we are at
    kx = (j * np.pi) / L
    ky = (l * np.pi) / (q * L)
    return np.cos(kx * x) * np.cos(ky * y)


#Create a 128x128 grid and initialise a near homogeneous steady state with a small random perturbation. Use the parameters a = 3, b = 8, Du = 1, Dv = 2.3, and time step = 0.01. Plot the solution at times t = 0, 100, 200, 300, and 400.
u01, v01 = initiation(L, u, v)



#Discretize the Laplacian


vmin = 0
vmax = 12.2

for Dv in Dvs:
    u = u01.copy()
    v = v01.copy()
    u_old = u.copy()
    steps_steady = 0
    for t in range(n_steps):
        new_u = laplacian(u)
        new_v = laplacian(v)

        
        dudt = a - (b+1) * u + u**2 * v + Du * new_u
        dvdt = b * u - u**2 * v + Dv * new_v

        u += dudt * time_step
        v += dvdt * time_step
    
        if t == 1000:
            sns.heatmap(u, cmap = "viridis", vmin=vmin, vmax=vmax)
            plt.title(f"u at time = {t} for Dv = {Dv}")
            plt.show()
        if t >= 1000 and np.max(np.abs(u - u_old)) < 1e-3:
            steps_steady += 1
            if steps_steady >= 50:
                sns.heatmap(u, cmap = "viridis", vmin=vmin, vmax=vmax)
                plt.title(f"u at time = {t} for Dv = {Dv}")
                plt.show()
                break
        elif np.max(np.abs(u - u_old)) >= 1e-3:
            steps_steady = 0
        u_old = u.copy()










#Implement periodic boundary conditions

#Time-step the PDE using delta t = 0.01

#Simulate 

#Plot heatmaps of u, after 1000 iterations and after steady state


