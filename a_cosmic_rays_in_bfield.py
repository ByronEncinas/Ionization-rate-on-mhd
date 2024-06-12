# import for visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from z_library import *

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
if True:
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

"""### Constants and Parameters"""

"""  
Unstructured X, Y, Z Mesh Grid
"""

scale_factor = 0.0 
relativepos = [x[0][0][1]-x[0][0][0], y[0][0][1]-y[0][0][0], z[0][0][1]-z[0][0][0]]
scale_factor += magnitude(relativepos)
relativepos = [x[0][1][0]-x[0][0][0], y[0][1][0]-y[0][0][0], z[0][1][0]-z[0][0][0]]
scale_factor += magnitude(relativepos)
relativepos = [x[1][0][0]-x[0][0][0], y[1][0][0]-y[0][0][0], z[1][0][0]-z[0][0][0]]
scale_factor += magnitude(relativepos)
scale_factor /= 3.0 


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

if DS < 64:
  plot_3d_vec_field(Bx, By, Bz)

import random

ds = DS # Magnetic Field, Point in grid and Points that enclose it
margin = 3.0  # Adjust the margin as needed

POINT_i = 121.4
POINT_j = 109.1
POINT_k = 13.6

# Random Starting Point Y = 0
if False:
  POINT_i = random.uniform(margin, ds-margin)
  POINT_j = random.uniform(margin, ds-margin)
  POINT_k = random.uniform(margin, ds-margin)

xpos = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, x) 
ypos = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, y)
zpos = interpolate_scalar_field(POINT_i, POINT_j, POINT_k, z)

gridpos = np.array([xpos,ypos,zpos])
realprevpos = [0.0, 0.0, 0.0]

print("Point Generated is: ",[POINT_i, POINT_j, POINT_k])
print("Corresponds with: ", gridpos)

prev_gridpos = np.array([xpos,ypos,zpos])
run_prev_pos = np.array([POINT_i, POINT_j, POINT_k])
run_cur_pos = np.array([POINT_i, POINT_j, POINT_k])   # initial position

timestep = np.linspace(0, TOTAL_TIME, SNAPSHOTS + 1)  # start time, final_time, number of snapshots
delta = (timestep[1] - timestep[0])*10e+19                     # delta timestep

follown = [run_cur_pos.tolist()]                      # all trajectory points
Bpsn = [interpolate_vector_field(run_cur_pos[0], run_cur_pos[1], run_cur_pos[2], Bx, By, Bz)]  # all trajectory points

lin_seg_g = 0.0      
lin_seg_n = 0.0                                         # distance of path traveled (s)
bf_mag =  0.0                                         # magnetic field at s-distance

line_segment_g = []                                # acumulative path distances
bfield_magnitud_g = []  
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

CourantNumber = Bpsn[0]*delta/scale_factor

#os.mkdir(f"b_output_data/{starting_pos}_file")
#with open(f"b_output_data/{starting_pos}_file/initpos_{starting_pos}_iter.txt", "w") as run_data:
with open(f"a_output_data/critical_points.txt", "w") as run_data: #tests

        while True:
            try:
                # B Field at current position and save
                Bp_run = np.array(interpolate_vector_field(run_cur_pos[0], run_cur_pos[1], run_cur_pos[2], Bx, By, Bz))
                Bpsn.append(Bp_run)
                bf_mag = magnitude(Bp_run)

                # unit vector in field direction
                unito = Bp_run/bf_mag

                #auxp = rk4_int(CONST, run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], Bx, By, Bz, delta)
                run_cur_pos += unito*delta # s is equally spaced now
                
                xpos = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], x) 
                ypos = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], y)
                zpos = interpolate_scalar_field(run_cur_pos[0],run_cur_pos[1],run_cur_pos[2], z)

                gridpos += unito*delta
                #print(count, magnitude(run_cur_pos,run_prev_pos))
                follown.append(gridpos.tolist())
                           
                run_data.write(f"{count}, {lin_seg_n}, {xpos}, {ypos}, {zpos},{bf_mag},{Bp_run[0]},{Bp_run[1]},{Bp_run[2]},{run_cur_pos[0]},{run_cur_pos[1]},{run_cur_pos[2]}\n") 
            except:
                print("Particle got out of the Grid")
                break

            lin_seg_g +=  magnitude(gridpos, prev_gridpos)
            lin_seg_n +=  magnitude(run_cur_pos, run_prev_pos)*scale_factor # centimeters
            #print(magnitude(gridpos, prev_gridpos),magnitude(run_cur_pos, run_prev_pos)*scale_factor)
            run_prev_pos = run_cur_pos.copy()
            prev_gridpos = gridpos.copy()

            line_segment_n.append(lin_seg_n)
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
print("RunggeKutta4: ",follown[0], "-->", gridpos)

print("Min, Max of B in trayectory: ", min_diff_bp, max_diff_bp)

print(scale_factor)

"""# Graphs"""

''' print(len(line_segment_n),len(bfield_magnitud_n))  '''

np.linspace(0, TOTAL_TIME, SNAPSHOTS + 1)

pocket_finder(bfield_magnitud_n, plot=True)

if True:
  try:
    plot_trajectory_versus_magnitude(line_segment_n,bfield_magnitud_n, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])
    plot_trajectory_versus_magnitude(line_segment_g,bfield_magnitud_n, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])
    


  except Exception as e:
    print(e)
    print("Not Possible to plot")


