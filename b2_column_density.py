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

distance      = np.array(list(left_distance) + [f + d for d in right_distance[1:]])
radius_vector = np.array(list(left_radius_vector) + right_radius_vector[1:])
bfield        = np.array(list(left_bfield_magnitudes) + right_bfield_magnitudes[1:])

print("Simulation Successfull")

precision = 200

muforward   = np.array([k/(10) for k in range(1,11)]) # Forward Ionization
mubackward  = np.array([k/(10)-1 for k in range(0,10)]) # Backward Ionization

dmu = 1/precision

############----------------------------------FORWARD----------------------------------############

ds = abs(distance[1] - distance[0]) # they are equally space (unito * delta)

s = np.random.uniform(0,1,2)
#mu_ism    = np.flip(sorted(np.exp(-abs(s))))
mu_ism = np.flip(np.logspace(-3,0,1000))

np.save("PitchAngleCosines.npy", mu_ism)

c = (len(mu_ism), len(distance))
Nforward  = np.zeros(c)

B_ism     = bfield[0]

for i, mui_ism in enumerate(mu_ism):
    print("====================>  ",mui_ism)

    for j in range(len(distance)):
        
        gaspos    = radius_vector[j]  # Position for s in structured grid
        num_den_j = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density order of 1.0^0      
        Bsprime   = bfield[j]
        deno      = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        
        if deno < 0:
            """ 
            1 - (Bsprime/B_ism) * (1 - muj_ism**2) < 0
            """
            break
        
        delta_nj  = ds / np.sqrt(deno)
        Nforward[i,j] = Nforward[i,j-1] + num_den_j*delta_nj

np.save("ForwardColumn.npy", Nforward)



############---------------------------------BACKWARD----------------------------------############

ds = abs(distance[1] - distance[0]) # they are equally space (unito * delta)

c = (len(mu_ism), len(distance))
Nbackward  = np.zeros(c)
rev_radius_vector = radius_vector[::-1]
rev_bfield        = bfield[::-1]
B_ism     = bfield[-1]

for i, mui_ism in enumerate(mu_ism):
    mui_ism = -mui_ism
    print("====================> ", mui_ism)

    for j in range(len(distance)):
        
        gaspos    = rev_radius_vector[j]  # Position for s in structured grid
        num_den_j = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density order of 1.0^0      
        Bsprime   = rev_bfield[j]
        deno      = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        
        if deno < 0:
            """ 
            1 - (Bsprime/B_ism) * (1 - muj_ism**2) < 0
            """
            break
        
        delta_nj  = ds / np.sqrt(deno)
        Nbackward[i,j] = Nbackward[i,j-1] + num_den_j*delta_nj

np.save("BackwardColumn.npy", Nbackward)

if True:
    import matplotlib.cm as cm

    colors = cm.rainbow(np.linspace(0, 1, len(mu_ism)))

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Create a 2x1 grid of subplots

    for i, c in enumerate(colors):
        axs[0].scatter(distance/10e+18, Nforward[i,:], label=f'$N_+(\mu_i = {mu_ism[i]})$', linestyle="--", color=c, s =5)

    axs[0].set_yscale('log')
    #axs[0].set_xscale('log')
    #axs[0].set_title('Forward Column Densities')
    axs[0].set_ylabel('$log_10(N_+(\mu))$ (log scale)')
    #axs[0].legend()
    axs[0].grid(True)
    
    for i, c in enumerate(colors):
        axs[1].scatter(distance/10e+18, Nbackward[i,:], label=f'$N_-(\mu_i = -{mu_ism[i]})$', linestyle="--", color=c, s=5)

    axs[1].set_yscale('log')
    #axs[1].set_xscale('log')
    #axs[1].set_title('Backward Column Densities')
    axs[1].set_xlabel('Distance ($s$ $cm/10^{18}$)')
    axs[1].set_ylabel('$log_10(N_-(\mu_i))$ (log scale)')
    #axs[0].legend()
    axs[1].grid(True)
    
    
    axs[2].plot(mu_ism, label=f'$\mu_i-distribution$', linestyle=":", color="m")
    #axs[2].set_title('$\mu_i$ distribution')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('$\mu_i$ distribution')
    axs[2].legend()
    axs[2].grid(True)

    plt.savefig("b_output_data/Columns&MuiDistro.png")

    #plt.show()
