# import for visualization
import matplotlib.pyplot as plt
import json
from z_library import *
import numpy as np

"""  
Declaring and Importing Important Parameters

"""
gas_den = np.array(np.load("input_data/gas_number_density.npy", mmap_mode='r'))

if True: # import data colapse to see statements
    # magnetic field in [x,y,z]
    Bx = np.array(np.load("input_data/magnetic_field_x.npy", mmap_mode='r'))
    By = np.array(np.load("input_data/magnetic_field_y.npy", mmap_mode='r'))
    Bz = np.array(np.load("input_data/magnetic_field_z.npy", mmap_mode='r'))

    # using .npy data
    # Mesh Grid in Space
    x = np.array(np.load("input_data/coordinates_x.npy", mmap_mode='r'))
    y = np.array(np.load("input_data/coordinates_y.npy", mmap_mode='r'))
    z = np.array(np.load("input_data/coordinates_z.npy", mmap_mode='r'))

    # Velocity Dispersion
    # vel_disp = np.array(np.load("input_data/velocity_dispersion.npy", mmap_mode='r'))

    # this is temperature in [x,y,z]
    # temp = np.array(np.load("input_data/Temperature.npy", mmap_mode='r'))

    # Cosmic Ray Density
    # cr_den = np.array(np.load("input_data/cr_energy_density.npy", mmap_mode='r'))

    # Molecular Cloud Density
    # Ion Fraction
    # ion_frac = np.array(np.load("input_data/ionization_fraction.npy", mmap_mode='r'))

if True: # scale factor to convert grid index fractions into real distances in cm
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
    
# CONSTANT
point_i, point_j, point_k = int(), int(), int()
TOTAL_TIME = 9000000
TIMESTEP   = 0.05
SNAPSHOTS  = int(TOTAL_TIME/TIMESTEP)
DS         = 128
MARGIN     = 50

# this points contains several pocket, and it vanishes at the extremes.

point_i = 47.657
point_j = 81.482
point_k = 35.057

if False: # random point generator
    point_i = random.uniform(MARGIN, DS-MARGIN)
    point_j = random.uniform(MARGIN, DS-MARGIN)
    point_k = random.uniform(MARGIN, DS-MARGIN)

print(point_i, point_j, point_k)
# random point has been selected, now we gotta follow field lines

def trajectory(point_i, point_j, point_k, direction):    # this will be done only once in all simulation

    prev_pos = np.array([point_i, point_j, point_k])
    cur_pos = np.array([point_i, point_j, point_k])   # initial position

    timestep = np.linspace(0, TOTAL_TIME, SNAPSHOTS + 1)  # start time, final_time, number of snapshots
    delta = timestep[1] - timestep[0]                     # delta timestep

    if direction == -1:
        delta *= -1

    radius_vector = [cur_pos.tolist()]                      # all trajectory points

    bfield_s = [interpolate_vector_field( cur_pos[0],  cur_pos[1],  cur_pos[2], Bx, By, Bz)]  # all trajectory points

    lin_seg = 0.0                                         #distance of path traveled (s)
    bf_mag =  0.0                                         # magnetic field at s-distance

    distance = []                                # acumulative pathdistances
    bfield = []                             # magnetic field at each s-distance

    count = 0                                             # iterator count

    """# Calculating Trajectory"""
    # We are looking into the two nearest critical points, so let's look at points were first derivative 
    while all(ingrid(prev_pos[0], prev_pos[1], prev_pos[2])):
        try:
            # print(count, lin_seg ,bf_mag)
            # B Field at current position and save
            Bp_run = np.array(interpolate_vector_field(cur_pos[0], 
                                        cur_pos[1],  cur_pos[2], Bx, By, Bz))

            bfield_s.append(Bp_run)
            bf_mag = magnitude(Bp_run)

            # unit vector in field direction
            unito = Bp_run/bf_mag
            cur_pos += unito*delta
                        
            radius_vector.append(cur_pos)            
        except:

            print("Particle got out of the Grid")
            break

        lin_seg +=  magnitude(cur_pos,  prev_pos) * scale_factor # centimeters
        #print(magnitude(cur_pos,  prev_pos) * scale_factor)
        prev_pos = cur_pos.copy()

        distance.append(lin_seg)
        bfield.append(bf_mag)

        count += 1

    return (distance, radius_vector, bfield)

left_distance, left_radius_vector, left_bfield_magnitudes = trajectory(point_i, point_j, point_k, -1)
right_distance, right_radius_vector, right_bfield_magnitudes = trajectory(point_i, point_j, point_k, 1)

# this [1:] cut is because both lists contain the initial point
f = max(left_distance)

distance = list(left_distance) + [f + d for d in right_distance[1:]]
radius_vector = list(left_radius_vector) + right_radius_vector[1:]
bfield        = list(left_bfield_magnitudes) + right_bfield_magnitudes[1:]

print("Simulation Successfull")

size = 200
precision = size

muforward   = np.array([1 - k/(10) for k in range(11)]) # Backward Ionization
mubackward  = np.array([k/(10)-1 for k in range(11)]) # Backward Ionization

dmu = 1/precision

############----------------------------------FORWARD----------------------------------############
ds = abs(distance[1] - distance[0]) # they are equally space (unito * delta)

forward_column_density_at_mu = []
stop = False

for mui in muforward: # from 1 to 0
    
    dcolumn_density = 0.0

    for i in range(len(distance)): # this        

        Bs                  = bfield[i]                      # Magnetic field strength at the interest point
        column_density_at_s = []
        
        for j in range(len(distance[:i])): # traverse s' along field lines until s
            Bsprime   = bfield[j]
            bdash     = Bsprime / Bs  # Ratio of magnetic field strengths
            deno      = 1 - bdash * (1 - mui**2)
            #print(i, j, deno, dcolumn_density)
            
            if deno < 0: 
                # this pitch angle cosine doesnt reach s so it wont affect the column density
                dcolumn_density = 0.0
                stop = True
                break
            gaspos    = radius_vector[i]  # Position for s in structured grid
            numb_den  = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density order of 1.0^0       

            one_over         = 1.0 / np.sqrt(deno)               # Reciprocal of the square root term
            dcolumn_density += numb_den * ds * one_over   # Accumulate the contribution to column density        
        
        column_density_at_s.append(dcolumn_density) # column_at_mu = [N_at_s0, N_at_s1, ...]

    forward_column_density_at_mu.append(column_density_at_s)

    print("===FORWARDS===============>",i, mui, column_density_at_s[-1])

    if stop == True:
        stop == False
        """ 
        if column density at s reaches a point s_j < s then the particle does not reach 
        that distance with its current \mu, and so the column at all further positions 
        is unreachable for particles with same \mu. So the loop must continue with the next value of mu
        """
        continue
    
    
    
stop = False

import json

# Specify the file path
file_path = f'forward_column_density_mu={len(forward_column_density_at_mu)}.json'

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(forward_column_density_at_mu, json_file)

del column_density_at_s

backward_column_density_at_mu = []
stop = False
rev_bfield = bfield[::-1]
rev_radius_vector = radius_vector[::-1]
rev_distance = distance[::-1]

for mui in mubackward: # from 1 to 0
    
    dcolumn_density = 0.0
    for i in range(len(distance)): # this        

        Bs                  = rev_bfield[i]                      # Magnetic field strength at the interest point
        column_density_at_s = []
        
        for j in range(len(distance[:i])): # traverse s' along field lines until s
            Bsprime   = rev_bfield[j]
            bdash     = Bsprime / Bs  # Ratio of magnetic field strengths
            deno      = 1 - bdash * (1 - mui**2)
            #print(i, j, deno, dcolumn_density)
            
            if deno < 0: 
                # this pitch angle cosine doesnt reach s so it wont affect the column density
                dcolumn_density = 0.0
                stop = True
                break
            gaspos    = rev_radius_vector[i]  # Position for s in structured grid
            numb_den  = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density order of 1.0^0       

            one_over         = 1.0 / np.sqrt(deno)               # Reciprocal of the square root term
            dcolumn_density += numb_den * ds * one_over   # Accumulate the contribution to column density        
        
        column_density_at_s.append(dcolumn_density) # the index at column_density_at_mu corresponds with muforward index for the same \mu
    
    print("===BACKWARDS===============>",i, mui, column_density_at_s[-1])
    
    backward_column_density_at_mu.append(column_density_at_s)
    if stop == True:
        stop == False
        """ 
        if column density at s reaches a point s_j < s then the particle does not reach 
        that distance with its current \mu, and so the column at all further positions 
        is unreachable for particles with same \mu. So the loop must continue with the next value of mu
        """
        continue

# Specify the file path
file_path = f'backward_column_density_mu={backward_column_density_at_mu}=.json'

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(backward_column_density_at_mu, json_file)

if True:

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create a 2x1 grid of subplots

    axs[0].plot(distance, forward_column_density_at_mu[1], label=f'$N_+(\mu = 0.9)$', linestyle=":", color="c")
    #axs[0].plot(distance, column_density_at_all_s_with_mu1, label=f'N($\mu = 1.0$)', linestyle=":", color="m")
    axs[0].set_yscale('log')
    axs[0].set_title('Forward Column Densities')
    axs[0].set_xlabel('Distance (s)')
    axs[0].set_ylabel('Column Density $log_10(N(\mu))$ (log scale)')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(rev_distance, backward_column_density_at_mu[1], label=f'$N_-(\mu = -0.9*)', linestyle=":", color="c")
    axs[1].set_yscale('log')
    axs[1].set_title('Backward Column Densities')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Column Density $log_10(N(\mu))$ (log scale)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(pad=5)
    plt.savefig("b_output_data/Columns.png")

    #plt.show()
