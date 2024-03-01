import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
"""  
Appendix Function created to access simulation data



"""

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
        iteration = int(parts[0])
        traj_distance = float(parts[1])
        initial_position = [float(parts[2:5][0]), float(parts[2:5][1]), float(parts[2:5][2])]
        field_magnitude = float(parts[5])
        field_vector = [float(parts[6:9][0]), float(parts[6:9][1]), float(parts[6:9][2])]
        posit_index = [float(parts[9:][0]), float(parts[9:][1]), float(parts[9:][2])]

        data_dict = {
            'iteration': int(iteration),
            'trajectory (s)': traj_distance,
            'Initial Position (r0)': initial_position,
            'field magnitude': field_magnitude,
            'field vector': field_vector,
            'indexes': posit_index
        }

        return data_dict
    else:
        return None

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
    # Calculate min and max values of B_Fields
    if 0.0 in B_Fields:
        B_Fields.remove(0)
    if 0.0 in line_segments:
        line_segments.remove(0)
    min_value = min(B_Fields)
    max_value = max(B_Fields)

    plt.figure(figsize=(15, 5))

    plt.xlabel(legends[2])  # Add an x-label to the axes.
    plt.ylabel(legends[1])   # Add a y-label to the axes.

    plt.scatter(line_segments, B_Fields, marker='+', label='traj')  # Use scatter plot with "+" markers

    plt.title(legends[0])       # Add a title to the axes.
    plt.legend([legends[1]])           # Add a legend.
    plt.grid()

    # Set axis limits based on min and max values
    plt.ylim(min_value, max_value)

    if save_path:
        # Save the plot as PNG or SVG
        plt.savefig(save_path, format='png')  # Use format='svg' for SVG format

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

def plot_simulation_data(df):
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))  # Set the figure size as per your preference

    # hide axes
    for axs in ax:
        axs.axis('off')

    table = ax[0].table(cellText=df.head().values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    # Adjust font size for better readability
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Adjust cell heights for better alignment
    table.auto_set_column_width(col=list(range(len(df.columns))))

    ax[0].set_aspect('equal', adjustable='box')  # Set the aspect ratio to make it square

    fig.tight_layout(pad=0)
    plt.show()

""" 
$ python -i 1 2 3 
>> import sys
>> sys.argv
['1', '2', '3']
"""

# Specify the file path
file_path = sys.argv[1] #'output_data\[106.3,0.0,75.9]_file\initpos_[106.3,0.0,75.9]_iter.txt'

# Displaying a message about reading from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line and create a list of dictionaries
data_list = [process_line(line) for line in lines[:] if process_line(line) is not None]

# Creating a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)
df_shortened = df.head()
print(df_shortened)

# Extracting data into separate lists for further analysis
itera, scoord, posit, xpos, ypos, zpos, field_v, bmag, field_x, field_y, field_z, index = [], [], [], [], [], [], [], [], [], [], [], []

for iter in data_list: # Data into variables
    itera.append(iter['iteration'])
    scoord.append(iter['trajectory (s)'])
    posit.append(iter['Initial Position (r0)'])
    xpos.append(iter['Initial Position (r0)'][0])
    ypos.append(iter['Initial Position (r0)'][1])
    zpos.append(iter['Initial Position (r0)'][2])
    field_v.append(iter['field vector'])
    bmag.append(iter['field magnitude'])
    field_x.append(iter['field vector'][0])
    field_y.append(iter['field vector'][1])
    field_z.append(iter['field vector'][2])
    index.append(iter['indexes'])

#plot_simulation_data(df_shortened)

print(len(scoord), scoord[0])

print(len(bmag),bmag[0])

try:
  plot_trajectory_versus_magnitude(scoord[1:], bmag[1:] , ["$B$ Field Magnitude in Path", "$B$-Magnitude (Gauss) ($cm^{-1/2} g^{1/2}s^{−1}$)", "s-coordinate (cm)"])
except Exception as e:
  print(e)
  print("Not Possible to plot")