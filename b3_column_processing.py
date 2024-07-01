import matplotlib.pyplot as plt
import json

# Assuming your JSON file is named 'data.json'
file_path = 'b_columns.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data into a Python list
    column_at = json.load(file)

column_density_at_all_s_with_mu0 = []
column_density_at_all_s_with_muh = []
column_density_at_all_s_with_mu1  = []

for index, columns in enumerate(column_at):

    if len(columns) == 3:
        column_density_at_all_s_with_mu0.append(columns[0])
        column_density_at_all_s_with_muh.append(columns[1])
        column_density_at_all_s_with_mu1.append(columns[2])
    elif len(columns) == 2:
        column_density_at_all_s_with_muh.append(columns[0])
        column_density_at_all_s_with_mu1.append(columns[1])
    else:
        column_density_at_all_s_with_mu1.append(columns[0])  

fig, axs = plt.subplots(1, 1, figsize=(10, 10))  # Create a 2x1 grid of subplots

axs.plot(column_density_at_all_s_with_mu0, label=f'N($\mu=0.0$)', linestyle="--", color="g")
axs.plot(column_density_at_all_s_with_muh, label=f'N($\mu = 0.5$)', linestyle="--", color="c")
axs.plot(column_density_at_all_s_with_mu1, label=f'N($\mu = 1.0$)', linestyle="-", color="m")
axs.set_title('Forward Column Densities')
axs.set_xlabel('Distance (s)')
axs.set_ylabel('Column Density $log_10(N(\mu))$ (log scale)')
axs.grid(True)
plt.show()