from z_library import *
import json

"""  
Using Margo Data

Analysis of reduction factor

$$N(s) 1 - \sqrt{1-B(s)/B_l}$$

Where $B_l$ corresponds with (in region of randomly choosen point) the lowest between the highest peak at both left and right.
where $s$ is a random chosen point at original 128x128x128 grid.

1.- Randomly select a point in the 3D Grid. 
2.- Follow field lines until finding B_l, if non-existent then change point.
3.- Repeat 10k times
4.- Plot into a histogram.

contain results using at least 20 boxes that contain equally spaced intervals for the reduction factor.

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

if True:
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
point_i, point_j, point_k = int(), int(), int()
TOTAL_TIME = 9000000
TIMESTEP   = 0.05
SNAPSHOTS  = int(TOTAL_TIME/TIMESTEP)
CONST      = 1.0e+3
DS         = 128
MARGIN     = 34

print()
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

print()
print("Size of array matrix: {Bx.shape[0]}x{By.shape[0]}x{Bz.shape[0]}\n")

red_factor = []

import random

"""# Calculating """
# flow control to repeat calculations in no peak situations
cycle = 0 

import sys

if len(sys.argv) >= 2:
    max_cycles = int(sys.argv[1])
    print("max cycles:", max_cycles)
else:
    max_cycles = 100
    print("max cycles:", max_cycles)

while cycle < max_cycles:
    # this points contains several pocket, and it vanishes at the extremes.
    """ 
    point_i = 47.657
    point_j = 81.482
    point_k = 35.057
     """
    if True:
        point_i = random.uniform(MARGIN, DS-MARGIN)
        point_j = random.uniform(MARGIN, DS-MARGIN)
        point_k = random.uniform(MARGIN, DS-MARGIN)

    # random point has been selected, now we gotta follow field lines

    def trajectory(point_i, point_j, point_k, direction):    

        prev_pos = np.array([point_i, point_j, point_k])
        forward_cur_pos = np.array([point_i, point_j, point_k])   # initial position

        timestep = np.linspace(0, TOTAL_TIME, SNAPSHOTS + 1)  # start time, final_time, number of snapshots
        delta = timestep[1] - timestep[0]                     # delta timestep

        if direction == -1:
            delta *= -1

        radius_vector = [forward_cur_pos.tolist()]                      # all trajectory points

        bfield_s = [interpolate_vector_field( forward_cur_pos[0],  forward_cur_pos[1],  forward_cur_pos[2], Bx, By, Bz)]  # all trajectory points

        lin_seg = 0.0                                         #distance of path traveled (s)
        bf_mag =  0.0                                         # magnetic field at s-distance

        distance = []                                # acumulative pathdistances
        bfield = []                             # magnetic field at each s-distance

        count = 0                                             # iterator count

        """# Calculating Trajectory"""
        # We are looking into the two nearest critical points, so let's look at points were first derivative 
        for time in timestep:
            try:
                    # print(count, lin_seg ,bf_mag)
                    # B Field at current position and save
                    Bp_run = np.array(interpolate_vector_field(forward_cur_pos[0], 
                                        forward_cur_pos[1],  forward_cur_pos[2], Bx, By, Bz))
                    bfield_s.append(Bp_run)
                    bf_mag = magnitude(Bp_run)

                    # unit vector in field direction
                    unito = Bp_run/bf_mag
                    forward_cur_pos += unito*delta
                        
                    radius_vector.append(forward_cur_pos.tolist())
                        
            except:
                print("Particle got out of the Grid")
                break

            lin_seg +=  magnitude(forward_cur_pos,  prev_pos) * scale_factor # centimeters
            prev_pos =  forward_cur_pos.copy()

            distance.append(lin_seg)
            bfield.append(bf_mag)

            count += 1
        
        return (radius_vector, bfield)

    left_radius_vector, left_bfield_magnitudes = trajectory(point_i, point_j, point_k, -1)
    right_radius_vector, right_bfield_magnitudes = trajectory(point_i, point_j, point_k, 1)

     # this [1:] cut is because both lists contain the initial point
    radius_vector = list(reversed(left_radius_vector)) + right_radius_vector[1:]
    bfield        = list(reversed(left_bfield_magnitudes)) + right_bfield_magnitudes[1:]

    #print()
    #print(f"Initial Position: {forward_cur_pos}, Timestep: {delta}\n")
    #print("Integration: from ", radius_vector[0], "--> to ", radius_vector[-1])

    #print("Min, Max of B in trayectory: ", min_bp, max_bp)


    """# Obtained position along the field lines, now we find the pocket"""

    #index_peaks, global_info = pocket_finder(bfield) # this plots
    pocket, global_info = visualize_pockets(bfield, cycle, plot=False) # this plots
    index_pocket, field_pocket = pocket[0], pocket[1]

    # we can evaluate reduction factor if there are no pockets
    if len(index_pocket) < 2:
        # it there a minimum value of peaks we can work with? yes, two
        continue

    #print("")
    #print("index pocket: ", index_pocket)
    #print("field pocket: ", field_pocket)

    globalmax_index = global_info[0]
    globalmax_field = global_info[1]

    # Calculate the range within the 80th percentile
    start_index = len(bfield) // 10  # Skip the first 10% of indices
    end_index = len(bfield) - start_index  # Skip the last 10% of indices

    # we gotta find peaks in the interval   (B_l < random_element < B_h)
    # Generate a random index within the range
    p_r = random.randint(index_pocket[0], index_pocket[-1])
    #s_r = distance[p_r]
    B_r = bfield[p_r]

    print("random index: ", p_r, "peak's index: ", index_pocket)
    

    """How to find index of Bl?"""

    # Bl it is definitely between two peaks, we need to verify is also inside a pocket
    # such that Bl < Bs < Bh (p_i < p_r < p_j)

    # finds index at which to insert p_r and be kept sorted
    p_i = find_insertion_point(index_pocket, p_r)

    #print()
    print("Random Index:", p_r, "assoc. B(s_r):",B_r)
    print(index_pocket, p_i)

    if p_i is not None:
        # If p_i is not None, select the values at indices p_i-1 and p_i
        closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
    else:
        # If p_i is None, select the two closest values based on some other criteria
        continue

    if len(closest_values) == 2:
        B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
        B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
    else:
        continue

    if B_r/B_l < 1:
        R = 1 - np.sqrt(1-B_r/B_l)
        red_factor.append(R)
        cycle += 1
    else:
        continue
    
    print("Closest local maxima 'p':", closest_values)
    print("Bs: ", bfield[p_r], "Bi: ", bfield[closest_values[0]], "Bj: ", bfield[closest_values[1]])
    try:
        print("Bl: ", B_l, " B_r/B_l =", B_r/B_l, "< 1 ") 
    except:
        # this statement won't reach cycle += 1 so the cycle will continue again.
        continue
    """ 
    bs: where bs is the field magnitude at the random point chosen 
    bl: magnetic at position s of the trajectory
    """
    
import json

# Specify the file path
file_path = 'prev_meeting_random_distributed_reduction_factor.json'

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(red_factor, json_file)


print(red_factor)
"""# Graphs"""

# shape of field lines 
print("Plot B(s)")

#plot_trajectory_versus_magnitude(distance, bfield, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])

# Here we plot the histogram for given reduction factor
import matplotlib.pyplot as plt

if len(sys.argv) >= 2:
    bins = int(sys.argv[2])
    print("bins:", bins)
else:
    bins = 100
    print("bins:", bins)

bins = 100
# Plotting the histogram
plt.hist(red_factor, bins=bins, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel(f'bins (bins)')
plt.ylabel('Reduction Factor (R)')
plt.title('Reduction Factor across randomized selection of points in MC')

plt.savefig(f"c_output_data/histogram{bins}.png")

# Displaying the histogram
#plt.show()
