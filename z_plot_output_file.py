import pandas as pd
import re

"""  
Appendix Function created to access simulation data



"""
# Function to extract values from the line
def extract_values(line):
    return [float(match.group()) for match in re.finditer(r'\d+\.\d+', line)]

# Function to process each line of the file
def process_line(line):
    parts = line.split(', ')
    if len(parts) > 1:
        # {iteration}, {trajectory_line}, {position_x}, {position_y}, {position_y},{field_magnitude},{field_x},{field_y},{field_z
        print(parts[2:5])
        iteration = int(parts[0])
        traj_distance = parts[1]
        initial_position = parts[2:4]
        timestep = float(extract_values(parts[2])[0])
        traj_data = extract_values(parts[3])

        data_dict = {
            'trajectory (s)': traj_distance,
            'Initial Position (r0)': initial_position,
            'Timestep': timestep,
            'Traj. Data': traj_data
        }

        return data_dict
    else:
        return None

# Specify the file path
file_path = 'output_data\[106.3,0.0,75.9]_file\initpos_[106.3,0.0,75.9]_iter.txt'

# Read the text file into a list of lines
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line and create a list of dictionaries
data_list = [process_line(line) for line in lines[2:] if process_line(line) is not None]

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)

# Display the DataFrame
print(df)
