from z_library import *
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
# Reduction Factor Along Field Lines (R vs S)

"""

scale_factor = 2.897279713439713e+19

def process_line(line):
    """
    Process a line of data containing information about a trajectory.

    Args:
        line (str): Input line containing comma-separated values.

    Returns:
        dict or None: A dictionary containing processed data if the line is valid, otherwise None.
    """
    parts = line.split(',')

    if len(parts) > 1:

        # c1_data.write(f"{count}, {lin_seg},{bf_mag},{Bp_run[0]},{Bp_run[1]},{Bp_run[2]},{cur_pos[0]},{cur_pos[1]},{cur_pos[2]}\n") 
        iteration = int(parts[0])
        traj_distance = float(parts[1])
        field_magnitude = float(parts[2])

        data_dict = {
            'iteration': int(iteration),
            'trajectory': traj_distance,
            'field magnitude': field_magnitude,
        }

        return data_dict
    else:
        return None

def pocket_finder(bmag, cycle=0, plot =False):
    """  
    j = i+1 > i
    """
    Bj = bmag[0]
    Bi = 0.0
    lindex = []
    lpeaks = []
    for B in bmag:
        Bj = B
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1] ): # if True, then we have a peak
            index = np.where(bmag==Bi)[0][0]
            lindex.append(index)
            lpeaks.append(Bi)
        Bi = B
    Bj = bmag[-1]
    Bi = 0.0
    rindex = []
    rpeaks = []
    for B in reversed(bmag):
        Bj = B
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]): # if True, then we have a peak
            index = np.where(bmag==Bi)[0][0]
            rindex.append(index)
            rpeaks.append(Bi)
        Bi = B
    peaks = lpeaks + list(reversed(rpeaks))[1:]
    indexes = lindex+list(reversed(rindex))[1:] 
    baseline = min(bmag)
    upline = max(bmag)
    index_global_max = np.where(bmag==upline)[0][0]
    
    if plot:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.plot(bmag)
        axs.plot(indexes,peaks, "x", color="green")
        axs.plot(indexes,peaks, ":", color="green")
        axs.plot(index_global_max, upline, "x", color="black")
        axs.plot(np.ones_like(bmag)*baseline, "--", color="gray")
        axs.set_xlabel("Index")
        axs.set_ylabel("Field")
        axs.set_title("Actual Field Shape")
        axs.legend(["bmag", "all peaks", "index_global_max", "baseline"])
        axs.grid(True)


        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"c_output_data/c_output_field_shape{cycle}.png")

        # Show the plot
        plt.show()
    return (indexes, peaks), (index_global_max, upline)

# Specify the file path
file_path = sys.argv[1]

    # Displaying a message about reading from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

    # Process each line and create a list of dictionaries
data_list = [process_line(line) for line in lines[:] if process_line(line) is not None]

    # Creating a DataFrame from the list of dictionaries
    #df = pd.DataFrame(data_list)

    # Extracting data into separate lists for further analysis
itera, distance, posit, field_v = [], [], [], []
bfield, field_x, field_y, field_z =  [], [], [], []

for iter in data_list: # Data into variables
    itera.append(iter['iteration'])
    distance.append(iter['trajectory'])
    bfield.append(iter['field magnitude'])

itera = np.array(itera)
distance = np.array(distance)
bfield = np.array(bfield)

print(" Data Successfully Loaded")

reduction_factor_at_s = []
inv_reduction_factor_at_s = []
normalized_bfield = []

pocket, global_info = pocket_finder(bfield, "_c1", plot=False) # this plots
index_pocket, field_pocket = pocket[0], pocket[1]

global_max_index = global_info[0]
global_max_field = global_info[1]

print(index_pocket)
print(field_pocket)

#################################################333

for i, Bs in enumerate(bfield): 
    """  
    R = 1 - \sqrt{1 - B(s)/Bl}
    s = distance traveled inside of the molecular cloud (following field lines)
    Bs= Magnetic Field Strenght at s
    """
    if i < index_pocket[0] or i > index_pocket[-1]: # assuming its monotonously decreasing at s = -infty, infty
        Bl = Bs
    
    p_i = find_insertion_point(index_pocket, i)    
    indexes = index_pocket[p_i-1:p_i+1]       

    if len(indexes) < 2:
        nearby_field = [Bs, Bs]
    else:
        nearby_field = [bfield[indexes[0]], bfield[indexes[1]]]

    Bl = min(nearby_field)

    if Bs/Bl < 1:
        R = 1 - np.sqrt(1. - Bs/Bl)
    else:
        R = 1

    print(i, p_i, Bs/Bl, R)
    reduction_factor_at_s.append(R)
    inv_reduction_factor_at_s.append(1/R)
    normalized_bfield.append(Bs/global_max_field-0.01)

##############################################33

if True:
    # Create a 2x1 subplot grid
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Scatter plot for Case Zero
    axs[0].plot(reduction_factor_at_s, label='$R(s) = 1 - \sqrt{1-B(s)/B_l}$', linestyle='--', color='grey')
    axs[0].set_ylabel('Reduction Factor $R(s)$')
    axs[0].set_xlabel('$s-distance$')
    axs[0].legend()

    # Scatter plot for Case Zero
    axs[1].plot(inv_reduction_factor_at_s, label='$1/R(s) \ Inverse Reduction Factor$', linestyle='--', color='gray')
    axs[1].plot(normalized_bfield, label='$\hat{B}(s) \ Normalized Field Profile$', linestyle='-', color='black')
    axs[1].set_xlabel('$s-distance$')
    axs[1].legend()

    axs[2].plot(bfield, label='$B(s)$', linestyle='-', color='r')
    axs[2].set_ylabel('B(s) Field Strength$')
    axs[2].set_xlabel('$s-distance (cm)$')
    axs[2].legend()

    # Adjust layout
    #plt.tight_layout()

    # Save Figure
    plt.savefig("reductionf_at_s.png")

    # Display the plot
    #plt.show()
