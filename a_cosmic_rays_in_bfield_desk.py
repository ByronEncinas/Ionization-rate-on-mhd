# import for visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""# Objective

**This code is intended to solve for the trajectory of a particle following magnetic field lines.**

The procedure is as follows:

1. Setting up initial conditions

We set up the initial position inside the grid for the particle.

<code>
pi, pj, pk = initialx, initialy, initialz
</code>

2. Finding the 8 Magnetic field vectors that enclose the initial position

We use the function <code>interpolate_vector_field(pi, pj, pk)</code> &  <code>find_enclosing_vectors(grid, i, j, k)</code> to facilitate the interpolation in unknown points inside of the grid.

We follow Magnetic Field Lines (guiding centers) for charged particles.

# Importing Data
"""

# using .npy data
# this is temperature in [x,y,z]
temp = np.array(np.load("input_data/Temperature.npy", mmap_mode='r'))

# magnetic field in [x,y,z]
Bx = np.array(np.load("input_data/magnetic_field_x.npy", mmap_mode='r'))
By = np.array(np.load("input_data/magnetic_field_y.npy", mmap_mode='r'))
Bz = np.array(np.load("input_data/magnetic_field_z.npy", mmap_mode='r'))

# Cosmic Ray Density
cr_den = np.array(np.load("input_data/cr_energy_density.npy", mmap_mode='r'))

# Molecular Cloud Density
gas_den = np.array(np.load("input_data/gas_number_density.npy", mmap_mode='r'))

# Ion Fraction
ion_frac = np.array(np.load("input_data/ionization_fraction.npy", mmap_mode='r'))

# Mesh Grid in Space (Unstructured)
x = np.array(np.load("input_data/coordinates_x.npy", mmap_mode='r'))
y = np.array(np.load("input_data/coordinates_y.npy", mmap_mode='r'))
z = np.array(np.load("input_data/coordinates_z.npy", mmap_mode='r'))

# Velocity Dispersion
vel_disp = np.array(np.load("input_data/velocity_dispersion.npy", mmap_mode='r'))


def magnitude(new_vector, prev_vector=[0.0,0.0,0.0]): 
    return np.sqrt(sum([(new_vector[i]-prev_vector[i])*(new_vector[i]-prev_vector[i]) for i in range(len(new_vector))]))

"""  
Unstructured X, Y, Z Mesh Grid

# X Positions
print(x[0][0][0], x[0][0][1], x[0][1][0], x[0][1][1])
print(x[1][0][0], x[1][0][1], x[1][1][0], x[1][1][1], '\n')

# Y Positions
print(y[0][0][0], y[0][0][1], y[0][1][0], y[0][1][1])
print(y[1][0][0], y[1][0][1], y[1][1][0], y[1][1][1], '\n')

# Z Positions
print(z[0][0][0], z[0][0][1], z[0][1][0], z[0][1][1])
print(z[1][0][0], z[1][0][1], z[1][1][0], z[1][1][1], '\n')

# Position of vector from (0,0,0) to (0,0,1)
relativepos = [x[0][0][1]-x[0][0][0], y[0][0][1]-y[0][0][0], z[0][0][1]-z[0][0][0]]
print(relativepos)

relativepos = [x[0][1][0]-x[0][0][0], y[0][1][0]-y[0][0][0], z[0][1][0]-z[0][0][0]]
print(relativepos)

relativepos = [x[1][0][0]-x[0][0][0], y[1][0][0]-y[0][0][0], z[1][0][0]-z[0][0][0]]
print(relativepos)

# scale factor to change from unitary cells to real space
scale_factor = 

Result:
2.791736544372304e+19 2.7919206447928984e+19 2.791264591149962e+19 2.791302103840296e+19
2.8206573324639883e+19 2.8208470432343945e+19 2.820246631289088e+19 2.8206338937783726e+19 

2.7923448990381445e+19 2.791320227977935e+19 2.821533929948694e+19 2.820783752917449e+19
2.7920766519432114e+19 2.791083131810591e+19 2.820975513600431e+19 2.820253222972481e+19 

2.7914320208369213e+19 2.8202072708803523e+19 2.7908040214520828e+19 2.818972113194587e+19
2.7918259321903374e+19 2.820169281646712e+19 2.7909315799961674e+19 2.819254091862435e+19 

[ 1841004205944832.0,     -1.024671060209664e+16,   2.8775250043430912e+17]
[-4719532223418368.0,      2.9189030910549606e+17, -6279993848385536.0    ]
[ 2.8920788091684454e+17, -2682470949330944.0,      3939113534160896.0    ]

"""

scale_factor = 0.0 

relativepos = [x[0][0][1]-x[0][0][0], y[0][0][1]-y[0][0][0], z[0][0][1]-z[0][0][0]]
print(relativepos)
scale_factor += magnitude(relativepos)

relativepos = [x[0][1][0]-x[0][0][0], y[0][1][0]-y[0][0][0], z[0][1][0]-z[0][0][0]]
print(relativepos)

scale_factor += magnitude(relativepos)

relativepos = [x[1][0][0]-x[0][0][0], y[1][0][0]-y[0][0][0], z[1][0][0]-z[0][0][0]]
print(relativepos)
scale_factor += magnitude(relativepos)

scale_factor /= 3.0 

"""### Constants and Parameters"""

global TOTAL_TIME, SNAPSHOTS, TIMESTEP

# CONSTANT
POINT_i, POINT_j, POINT_k = int(), int(), int()
TOTAL_TIME = 9_000_000
TIMESTEP   = 0.05
SNAPSHOTS  = int(TOTAL_TIME/TIMESTEP)
CONST      = 1.0e+3
DS         = 128
MARGIN     = 3.0

print("Timestep Delta: ", TIMESTEP)
print("Total Time    : ", TOTAL_TIME)
print("Snapshots     : ", SNAPSHOTS)

# We cut the size of data for faster processing
Bx = Bx[0:DS, 0:DS, 0:DS]
By = By[0:DS, 0:DS, 0:DS]
Bz = Bz[0:DS, 0:DS, 0:DS]
x =   x[0:DS, 0:DS, 0:DS]
y =   y[0:DS, 0:DS, 0:DS]
z =   z[0:DS, 0:DS, 0:DS]

print(f"\nSize of array matrix: {Bx.shape[0]}x{By.shape[0]}x{Bz.shape[0]}\n")

"""# Methods

##Simulation Functions

### Numerical Integration
"""

def Ind(i):
  '''
  Indicator Function
              1     if   0 < i < 128
  Ind(i) = {
              1     if   i < 0 or i > 128
  '''         
  # special case is ds = 128
  stat_i = 0<=i<=128

  if stat_i:
    return 1
  else:
    return 0

def four_point_derivative(f, x, h): 
    """
    Approximate the derivative of a function using the 4-point central difference formula.

    Parameters:
    - f: The function to differentiate.
    - x: The point at which to approximate the derivative.
    - h: Step size.
    """
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)

def eul_int(fderiv, dt):
    return fderiv * dt

def rk4_int(const, x, y, z,field_x,field_y,field_z,dt):
    '''
    fderiv: First Derivative  -> Velocity
    x: Position
    '''
    k_1, k_2, k_3, k_4 = 0.0, 0.0, 0.0, 0.0

    k_1 = const * interpolate_vector_field(x, y, z,field_x,field_y,field_z)
    k_2 = const * interpolate_vector_field(x+0.5*k_1[0]*dt, y+0.5*k_1[1]*dt, z + 0.5*k_1[2]*dt,field_x,field_y,field_z)
    k_3 = const * interpolate_vector_field(x+0.5*k_2[0]*dt, y+0.5*k_2[1]*dt, z + 0.5*k_2[2]*dt,field_x,field_y,field_z)
    k_4 = const * interpolate_vector_field(x+k_3[0]*dt, y+k_3[1]*dt, z + k_3[2]*dt,field_x,field_y,field_z)

    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

def run_second_order(const, position, velocity, Bx, By, Bz, timestep):
    '''
    fderiv: First Derivative  -> Velocity : np.array
    sderiv: First Derivative  -> Acceleration : np.array
    x,y,z: Position : np.array
    vx,vy,vz: Velocity : np.array
    timestep
    '''
    k_1, k_2, k_3, k_4 = 0.0, 0.0, 0.0, 0.0
    l_1,l_2,l_3,l_4 = 0.0, 0.0, 0.0, 0.0

    l_1, k_1 = const * interpolate_vector_field(position, Bx, By, Bz), velocity
    l_2, k_2 = const * interpolate_vector_field(position[0]+0.5*k_1[0]*timestep, position[1]+0.5*k_1[1]*timestep, position[2] + 0.5*k_1[2]*timestep, Bx, By, Bz), velocity + timestep*l_1/2
    l_3, k_3 = const * interpolate_vector_field(position[0]+0.5*k_2[0]*timestep, position[1]+0.5*k_2[1]*timestep, position[2] + 0.5*k_2[2]*timestep, Bx, By, Bz), velocity + timestep*l_2/2
    l_4, k_4 = const * interpolate_vector_field(position[0]+k_3[0]*timestep, position[1]+k_3[1]*timestep, position[2] + k_3[2]*timestep, Bx, By, Bz), velocity + timestep*l_3

    new_velocity = velocity + timestep * (l_1 + 2 * l_2 + 2 * l_3 + l_4) / 6
    new_position = position + timestep * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    return new_position, new_velocity

"""### Vector Functions"""

def find_enclosing_vectors(i, j, k):
    vectors = []
    neighbors = [
        (0, 0, 0),  # Vertex at the origin
        (0, 0, 1),  # Vertex on the x-axis
        (0, 1, 0),  # Vertex on the y-axis
        (0, 1, 1),  # Vertex on the z-axis
        (1, 0, 0),  # Vertex on the xy-plane
        (1, 0, 1),  # Vertex on the xz-plane
        (1, 1, 0),  # Vertex on the yz-plane
        (1, 1, 1)   # Vertex in all three dimensions
    ]

    point = [np.floor(i), np.floor(j), np.floor(k)]

    for neighbor in neighbors:
        ni, nj, nk = neighbor

        if 0 <= i + ni and 0 <= j + nj and 0 <= k + nk:
            dx = ni + i
            dy = nj + j
            dz = nk + k
            vectors.append(np.array([np.floor(dx), np.floor(dy), np.floor(dz)]))

    return vectors

def interpolate_vector_field(point_i, point_j, point_k, field_x, field_y, field_z):
    u, v, w = point_i - np.floor(point_i), point_j - np.floor(point_j), point_k - np.floor(point_k)

    # Find the eight vectors enclosing the chosen point
    cube = find_enclosing_vectors(point_i, point_j, point_k)

    # Initialize Bp
    interpolated_vector = np.array([0.0, 0.0, 0.0])

    vectors = []

    for c in cube:
        i, j, k = int(c[0]), int(c[1]), int(c[2])

        # Check if indices are within the bounds of the magnetic field arrays
        if 0 <= i < Bx.shape[0] and 0 <= j < By.shape[0] and 0 <= k < Bz.shape[0]:

            vectors.append(np.array([Bx[i][j][k], By[i][j][k], Bz[i][j][k]]))
            #print([Bx[i][j][k], By[i][j][k], Bz[i][j][k]])
        else:
            # Handle the case where indices are out of bounds (you may choose to do something specific here)
            print(f"Indices out of bounds: {i}, {j}, {k}")

    interpolated_vector =  (1 - u) * (1 - v) * (1 - w) * vectors[0] + \
         (1 - u) * w * (1 - v) * vectors[1] + \
         (1 - u) * v * (1 - w) * vectors[2] + \
         (1 - u) * v * w * vectors[3] + \
         u * (1-v) * (1 - w) * vectors[4] + \
         u * (1 - v) * w * vectors[5] + \
         (1 - w) * v * u * vectors[6] + \
         u * v * w * vectors[7]

    return interpolated_vector

def ingrid(i,j,k, index = None):
  '''
  Function finds out if give a ds value for the dimensions of a cube ds x ds x ds grid, a given point i,j,k is inside or outside.
  we want to obtain a boolean function that identifies if a particle is outside of the grid, and the element(s) that are out of bounds
  '''
  # ds = 128 always!
  # True = in the grid, False = out of grid
  stat_i = False if (i < 0 or i > 128) else True
  stat_j = False if (j < 0 or j > 128) else True
  stat_k = False if (k < 0 or k > 128) else True

  vector = [stat_i,stat_j,stat_k]
  if all(vector): # if in grid
    return vector # [True, True, True]
  else: # if not in grid
    index = [not comp for comp in vector] # if vector = [True, False, False], index = [False, True, True] so all out of grid components are True
    return index

#ingrid(0,0,0,128)

"""### Scalar Functions"""

def find_enclosing_scalars(i, j, k):
    scalars = []
    neighbors = [
        (0, 0, 0),  # Vertex at the origin
        (0, 0, 1),  # Vertex on the x-axis
        (0, 1, 0),  # Vertex on the y-axis
        (0, 1, 1),  # Vertex on the z-axis
        (1, 0, 0),  # Vertex on the xy-plane
        (1, 0, 1),  # Vertex on the xz-plane
        (1, 1, 0),  # Vertex on the yz-plane
        (1, 1, 1)   # Vertex in all three dimensions
    ]
    point = [np.floor(i), np.floor(j), np.floor(k)] # three co-ordinates

    for neighbor in neighbors: # let's save all scalars in the 6 co-ordinates
        ni, nj, nk = neighbor # this is a given point

        if 0 <= i + ni and 0 <= j + nj and 0 <= k + nk:
            dx = ni + i
            dy = nj + j
            dz = nk + k
            scalars.append(np.array([np.floor(dx), np.floor(dy), np.floor(dz)]))
    return scalars

def interpolate_scalar_field(p_i, p_j, p_k, scalar_field, epsilon=1e-10): # implementing Inverse Distance Weighting (Shepard's method)

    

    # Origin of new RF primed: O' = [0,0,0] but O = [p_i, p_j, p_k] in not primed RF.
    u, v, w = p_i - np.floor(p_i), p_j - np.floor(p_j), p_k - np.floor(p_k)

    # Find the eight coordinates enclosing the chosen point
    coordinates = find_enclosing_scalars(p_i, p_j, p_k)

    # Initialize interpolated_vector
    interpolated_scalar = 0.0
    cum_sum, cum_inv_dis= 0.0, 1.0

    if all(ingrid(p_i,p_j,p_k)):
      for coo in coordinates:
            # coordinates of cube vertices around point    
            i, j, k = int(coo[0])*Ind(int(coo[0])), int(coo[1])*Ind(int(coo[1])), int(coo[2])*Ind(int(coo[2])) # O' position

            # distance from point to vertice of cube
            distance = magnitude([i - p_i, j - p_j, k - p_k]) # real distance
            #print(distance)
          
            # base case
            if distance <= epsilon:
                #print(f"known scalar field at [{p_i},{p_j},{p_k}]")
                return scalar_field[int(p_i)][int(p_j)][int(p_k)]

            cum_sum += scalar_field[i][j][k]/ (distance + epsilon) ** 2
            #print(scalar_field[i][j][k],(distance + 1e-10) ** 2)
            cum_inv_dis += 1 / (distance + epsilon) ** 2 
    else:
       new_coords = [[c[0],c[1],c[2]] for c in coordinates if all(ingrid(c[0],c[1],c[2]))]
       index = ingrid(p_i,p_j,p_k)
       complementary = [[new_coords[im][0]+index[0],new_coords[im][1]++index[1],new_coords[im][2]+index[2]] for im, boo in enumerate(new_coords)]
       coordinates = complementary+new_coords
       for coo in coordinates:
            # coordinates of cube vertices around point    
            i, j, k = int(coo[0]), int(coo[1]), int(coo[2]) # O' position

            # distance from point to vertice of cube
            distance = magnitude([i - p_i, j - p_j, k - p_k]) # real distance          
            #print(scalar_field[i][j][k],(distance + 1e-10) ** 2,np.exp(-distance))
            
            # base case
            if distance <= epsilon:
                #print(f"known scalar field at [{p_i},{p_j},{p_k}]")
                return scalar_field[int(p_i)][int(p_j)][int(p_k)]
            
            cum_sum += scalar_field[i][j][k]*np.exp(-distance**(3)/8)/ (distance + epsilon) ** 2
            cum_inv_dis += 1 / (distance + epsilon) ** 2 
            
    interpolated_scalar = cum_sum / cum_inv_dis
    return interpolated_scalar

"""### Plotting Functions"""

# Plot Trajectory of particle along field lines

def plot_enclosing_dots(total_time, snaps, field_x, field_y, field_z, p_i, p_j, p_k):

  # Create a meshgrid for the spatial coordinates
  x, y, z = np.meshgrid(np.arange(field_x.shape[0]), np.arange(field_y.shape[1]), np.arange(field_z.shape[2]))

  # Create a 3D plot
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_facecolor('black')  # Set background color to black

  # Plotting the magnetic field components
  ax.quiver(x, y, z, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

  # Choose a point in the grid (replace with your desired coordinates)

  timestep = np.linspace(0,total_time, snaps + 1)
  Bp = np.array([0.0,0.0,0.0])
  pos = np.array([0.0,0.0,0.0])

  cube = find_enclosing_vectors(p_i, p_j, p_k)

  for dot in cube:

    ax.scatter(dot[0],dot[1],dot[2], c = 'blue', marker='o')

  ax.quiver(x,y,z, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

  #ax.scatter(xcomp,ycomp,zcomp, c = 'blue', marker='o')
  ax.scatter(p_i, p_j, p_k, c = 'black', marker='x')

  # Set labels and title with white text
  ax.set_xlabel('X', color='white')
  ax.set_ylabel('Y', color='white')
  ax.set_zlabel('Z', color='white')
  ax.set_title('Enclosed Point to Interpolate $B_p$', color='black')

  # Set tick color to white
  ax.tick_params(axis='x', colors='white')
  ax.tick_params(axis='y', colors='white')
  ax.tick_params(axis='z', colors='white')

  # Show the plot
  plt.show()

def plot_3d_vec_field(field_x, field_y, field_z):
    ds = field_x.shape[0]
    x_plot, y_plot, z_plot = np.meshgrid(np.arange(ds), np.arange(ds), np.arange(ds))

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')  # Set background color to black

    # Plotting the magnetic field components
    ax.quiver(x_plot, y_plot, z_plot, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

    # Set labels and title with white text
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title('3D Magnetic Field Visualization', color='black')

    # Set tick color to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Show the plot
    plt.show()

def plot_trajectory(TOTAL_TIME, SNAPSHOTS, POINT_i, POINT_j, POINT_k, field_x, field_y, field_z, k):

    # Create a meshgrid for the spatial coordinates
    x, y, z = np.meshgrid(np.arange(field_x.shape[0]), np.arange(field_y.shape[1]), np.arange(field_z.shape[2]))

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')  # Set background color to black

    # Plotting the magnetic field components
    ax.quiver(x, y, z, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

    timestep = np.linspace(0, TOTAL_TIME, SNAPSHOTS + 1)

    trajectory = np.array([[0.0, 0.0, 0.0]])  # Initialize as an empty list

    delta = timestep[1] - timestep[0]
    pos = np.array([POINT_i, POINT_j, POINT_k])

    for time in timestep:

        Bp = interpolate_vector_field(pos[0], pos[1], pos[2], field_x, field_y, field_z)
        pos += k * Bp * delta
        # Update trajectory using np.vstack
        trajectory = np.vstack((trajectory, pos))

    # Plot the trajectory in blue
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', marker='x')

    # Set labels and title with white text
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title('Trajectory Plot in 3D', color='white')

    # Set tick color to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Show the plot
    plt.show()

def multiplot_trajectory_versus_magnitude(domain, legends,  *f ):

    '''
    legends = ["title", "y-coord", "x-coord"]
    '''
    plt.figure(figsize=(15, 5))

    plt.xlabel(legends[2])  # Add an x-label to the axes.
    plt.ylabel(legends[1])   # Add a y-label to the axes.

    plt.scatter(domain, f[0], marker='+', label='B Field Mag')  # Use scatter plot with "+" markers
    plt.scatter(domain, f[1], marker='*', label='CRs Density')  # Use scatter plot with "+" markers

    plt.title(legends[0])       # Add a title to the axes.
    plt.legend([legends[1]])           # Add a legend.
    plt.grid()
    
    min_value = min(f[1])
    max_value = max(f[0]) 

    # Set axis limits based on min and max values
    plt.ylim(min_value, max_value)

    plt.show()

def plot_trajectory_versus_magnitude(line_segments, B_Fields, legends, save_path=None):
    '''
    legends = ["title", "y-coord", "x-coord"]
    '''
    plt.figure(figsize=(15, 5))

    plt.xlabel(legends[2])  # Add an x-label to the axes.
    plt.ylabel(legends[1])   # Add a y-label to the axes.

    plt.scatter(line_segments, B_Fields, marker='+', label='traj')  # Use scatter plot with "+" markers

    plt.title(legends[0])       # Add a title to the axes.
    plt.legend([legends[1]])           # Add a legend.
    plt.grid()
    
    # Calculate min and max values of B_Fields
    if 0.0 in B_Fields:
      B_Fields.remove(0)
    min_value = min(B_Fields)
    max_value = max(B_Fields)

    # Set axis limits based on min and max values
    plt.ylim(min_value, max_value)

    if save_path:
        # Save the plot as PNG or SVG
        plt.savefig(save_path, format='png')  # Use format='svg' for SVG format

    plt.show()

# Example usage:
# plot_trajectory_versus_magnitude(line_segments, B_Fields, legends, save_path='plot.png')

def plot_simulation_data(df):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout(pad = 0)
    plt.show()

#plot_simulation_data(data_frame)

"""# Context Plot"""

if DS < 64:
  plot_3d_vec_field(Bx, By, Bz)

"""# Setting Initial Conditions & Restrictions in Size"""

import random

ds = DS
# Magnetic Field, Point in grid and Points that enclose it
margin = 3.0  # Adjust the margin as needed

POINT_i = 82.0
POINT_j = 13.9
POINT_k = 12.8


# Random Starting Point Y = 0
if False:
  POINT_i = random.uniform(margin, ds-margin)
  POINT_j = random.uniform(margin, ds-margin)
  POINT_k = random.uniform(margin, ds-margin)

# 
xpos = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, x) 
ypos = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, y)
zpos = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, z)

realpos = [xpos,ypos,zpos]
realprevpos = [0.0, 0.0, 0.0]

print("Point Generated is: ",[POINT_i, POINT_j, POINT_k])
print("Corresponds with: ", realpos)

run_prev_pos = np.array([POINT_i, POINT_j, POINT_k])
run_cur_pos = np.array([POINT_i, POINT_j, POINT_k])   # initial position
run_cur_vel = np.array([0.0, 0.0, 0.0])               # initial position

timestep = np.linspace(0, TOTAL_TIME, SNAPSHOTS + 1)  # start time, final_time, number of snapshots
delta = timestep[1] - timestep[0]                     # delta timestep

follown = [run_cur_pos.tolist()]                      # all trajectory points

Bpsn = [interpolate_vector_field(run_cur_pos[0], run_cur_pos[1], run_cur_pos[2], Bx, By, Bz)]  # all trajectory points

lin_seg = 0.0                                         # distance of path traveled (s)
bf_mag =  0.0                                         # magnetic field at s-distance

line_segment_n = []                                # acumulative path distances
bfield_magnitud_n = []                             # magnetic field at each s-distance

cr_density = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, cr_den)
mc_density = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, gas_den)

cr_den_n = [cr_density]
mc_den_n = [mc_density]

count = 0                                             # iterator count
max_diff_bp = 0.0                                     # maximum magnetic field experienced
min_diff_bp = 0.0                                     # minimum magnetic field experienced

starting_pos = '['+str(round(run_cur_pos[0],1)) +',' + str(round(run_cur_pos[1],1))+',' + str(round(run_cur_pos[2],1))+']'
print("Initial Position: ", starting_pos)

"""# Calculating Trajectory"""
import os

#os.mkdir(f"c_output_data/{starting_pos}_file")
#with open(f"c_output_data/{starting_pos}_file/initpos_{starting_pos}_iter.txt", "w") as run_data:
with open(f"critical_points.txt", "w") as run_data: #tests

        for time in timestep:
            try:
                # B Field at current position and save
                Bp_run = np.array(interpolate_vector_field(run_cur_pos[0], run_cur_pos[1], run_cur_pos[2], Bx, By, Bz))
                Bpsn.append(Bp_run)
                bf_mag = magnitude(Bp_run)

                # unit vector in field direction
                unito = Bp_run/bf_mag

                #auxp = rk4_int(CONST, run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], Bx, By, Bz, delta)
                run_cur_pos += unito*delta # s is equally spaced now
                
                print(count, magnitude(run_cur_pos,run_prev_pos))
                follown.append(run_cur_pos.tolist())

                xpos = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], x) 
                ypos = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], y)
                zpos = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], z)

                realpos = [xpos,ypos,zpos]
                
                #follown.append(realpos)
                follown.append(run_cur_pos.tolist())

                cr_density = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], cr_den)
                mc_density = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], gas_den)

                cr_avg = (cr_density + cr_den_n[-1]) / 2.0
                mc_avg = (mc_density + mc_den_n[-1]) / 2.0

                cr_den_n.append(cr_avg)
                mc_den_n.append(mc_avg)
                
                run_data.write(f"{count}, {lin_seg}, {xpos}, {ypos}, {zpos},{bf_mag},{Bp_run[0]},{Bp_run[1]},{Bp_run[2]},{run_cur_pos[0]},{run_cur_pos[1]},{run_cur_pos[2]}\n") 

            except:

                print("Particle got out of the Grid")

                break

            lin_seg +=  magnitude(run_cur_pos, run_prev_pos)*scale_factor # centimeters
            run_prev_pos = run_cur_pos.copy()

            
            line_segment_n.append(lin_seg)
            bfield_magnitud_n.append(bf_mag)

            count += 1

bmag = bfield_magnitud_n.copy()

try:
  max_diff_bp = max(bmag)
  min_diff_bp = min(bmag)
except:
  exit()

print(f"Initial Position: {starting_pos}, Timestep: {delta}\n")
print("Initial Position Vector --> Final Position Vector")
print("RunggeKutta4: ",follown[0], "-->", run_cur_pos)

print("Min, Max of B in trayectory: ", min_diff_bp, max_diff_bp)

simulation_data = {
   "Starting Position": tuple(follown[0]),
   "Final Position": tuple(run_cur_pos),
   "Timestep (Delta t)": delta,
   "Method": "Runge Kutta 4, Second Order",
   "B Field Min, Max": (min_diff_bp, max_diff_bp),
   "Grid Size": f"{DS}x{DS}x{DS}"
}

"""# Scipy Interpolation for MC and CR data"""

from scipy.optimize import curve_fit
import numpy as np

# Your data
x = line_segment_n.copy()
y_mc = mc_den_n[1:].copy()
y_cr = cr_den_n[1:].copy()

# Define a fourth-degree polynomial function
def fourth_degree_polynomial(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Use curve_fit to fit the fourth-degree polynomial model to the data (for mc_den_n)
params_mc, covariance_mc = curve_fit(fourth_degree_polynomial, x, y_mc)

# Extract the parameters for mc_den_n
a_mc, b_mc, c_mc, d_mc, e_mc = params_mc

# Generate some points for the fitted fourth-degree polynomial curve (for mc_den_n)
x_fitted = np.linspace(min(x), max(x), len(x))
y_mc_fitted = fourth_degree_polynomial(x_fitted, a_mc, b_mc, c_mc, d_mc, e_mc)

# Use curve_fit to fit the fourth-degree polynomial model to the data (for cr_den_n)
params_cr, covariance_cr = curve_fit(fourth_degree_polynomial, x, y_cr)

# Extract the parameters for cr_den_n
a_cr, b_cr, c_cr, d_cr, e_cr = params_cr

# Generate some points for the fitted fourth-degree polynomial curve (for cr_den_n)
y_cr_fitted = fourth_degree_polynomial(x_fitted, a_cr, b_cr, c_cr, d_cr, e_cr)

print(scale_factor)
"""# Graphs"""

''' print(len(line_segment_n),len(bfield_magnitud_n))  '''

try:
  plot_trajectory_versus_magnitude(line_segment_n,bfield_magnitud_n, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])
  plot_trajectory_versus_magnitude(line_segment_n,mc_den_n[1:], ["Molecular Cloud Density in Path", "MC Density", "s-coordinate"])
  plot_trajectory_versus_magnitude(line_segment_n,cr_den_n[1:], ["Cosmic Ray Density in Path", "CR Density", "s-coordinate"])
  
  # Create a 1x3 subplot grid
  fig, axs = plt.subplots(3, 1, figsize=(10, 15))

  # Scatter plot for Case Zero
  axs[0].scatter(line_segment_n,bfield_magnitud_n, label='$B(s)$', marker='x', color='blue')
  
  axs[1].scatter(line_segment_n,mc_den_n[1:], label='$Cloud$', marker='+', color='green')
  axs[1].plot(x_fitted,y_mc_fitted, label='$Cloud$', linestyle='--', color='black')

  axs[2].scatter(line_segment_n,cr_den_n[1:], label='$CRs$', marker='*', color='red')
  axs[2].plot(x_fitted, y_cr_fitted, label='$CRs$', linestyle='--', color='black')


  axs[0].set_xlabel('$Log_{10}(E \ eV)$')
  axs[0].set_ylabel('$Log_{10}(J eV^{-1} cm^{-2} s^{-1} sr^{-1})$')

  axs[1].set_xlabel('$distance (cm)$')
  axs[1].set_ylabel('$MC Density gr/cm^3$')

  axs[2].set_xlabel('$distance (cm)$')
  axs[2].set_ylabel('$CR Density gr/cm^3$')

  # Add legends to each subplot
  axs[0].legend()
  axs[1].legend()
  axs[2].legend()

  # Adjust layout for better spacing
  #plt.tight_layout()

  # Save Figure
  plt.savefig(f"c_output_data/{starting_pos}_file/Svs_B_MC_CR_.png")

  # Display the plot
  plt.show()


except Exception as e:
  print(e)
  print("Not Possible to plot")

print(f"c_output_data/{starting_pos}_file/initpos_{starting_pos}_iter.txt")

