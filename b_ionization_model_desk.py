# import for visualization

import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd

"""  
Methods
"""

if True: # import data colapse to see statements
    # using .npy data
    # Mesh Grid in Space
    x = np.array(np.load("input_data/coordinates_x.npy", mmap_mode='r'))
    y = np.array(np.load("input_data/coordinates_y.npy", mmap_mode='r'))
    z = np.array(np.load("input_data/coordinates_z.npy", mmap_mode='r'))

    # Velocity Dispersion
    vel_disp = np.array(np.load("input_data/velocity_dispersion.npy", mmap_mode='r'))

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

def magnitude(new_vector, prev_vector=[0.0, 0.0, 0.0]):
    """
    Calculate the magnitude of a vector.
    
    Args:
        new_vector (list): The new vector.
        prev_vector (list): The previous vector. Default is [0.0, 0.0, 0.0].
        
    Returns:
        float: The magnitude of the vector.
    """
    import numpy as np
    return np.sqrt(sum([(new_vector[i] - prev_vector[i]) ** 2 for i in range(len(new_vector))]))

def eul_int(fderiv, timestep):
    """
    Perform Euler integration step.
    
    Args:
        fderiv (float): First derivative.
        timestep (float): Time step.
        
    Returns:
        float: Result of the Euler integration step.
    """
    return fderiv * timestep

def run_next_step(x, y, z, scalar_f, timestep):
    '''
    Perform a numerical integration step using the Runge-Kutta method.

    Args:
        x (float): X position.
        y (float): Y position.
        z (float): Z position.
        scalar_f: Scalar field.
        timestep (float): Time step.

    Returns:
        float: Result of the numerical integration step.
    '''
    k_1, k_2, k_3, k_4 = 0.0, 0.0, 0.0, 0.0

    k_1 = interpolate_scalar_field(x, y, z, scalar_f)
    k_2 = interpolate_scalar_field(x+0.5*k_1*timestep, y+0.5*k_1*timestep, z + 0.5*k_1*timestep, scalar_f)
    k_3 = interpolate_scalar_field(x+0.5*k_2*timestep, y+0.5*k_2*timestep, z + 0.5*k_2*timestep, scalar_f)
    k_4 = interpolate_scalar_field(x+k_3*timestep, y+k_3*timestep, z + k_3*timestep, scalar_f)

    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

def Ind(i):
  '''
    Indicator Function
    Function finds out if given a ds value for the dimensions of a cube ds x ds x ds grid, a given point i,j,k is inside or outside.

    Args:
        i (int): Dimension value.

    Returns:
        int: 1 if inside, 0 if outside.
  '''
  stat_i = 0<=i<=128

  if stat_i:
    return 1
  else:
    return 0

def ingrid(i,j,k, index = None):
  '''
    Function finds out if given a ds value for the dimensions of a cube ds x ds x ds grid, a given point i,j,k is inside or outside.
    we want to obtain a boolean function that identifies if a particle is outside of the grid, and the element(s) that are out of bounds.

    Args:
        i (int): X-coordinate.
        j (int): Y-coordinate.
        k (int): Z-coordinate.
        index: Not used in this function.

    Returns:
        list: [True, True, True] if inside, [False, True, True] if outside of grid in corresponding dimension(s).
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

def find_enclosing_scalars(i, j, k): # implementing Inverse Distance Weighting (Shepard's method)
    '''
    Find scalar values enclosing a given point in a cube grid.

    Args:
        i (float): X-coordinate.
        j (float): Y-coordinate.
        k (float): Z-coordinate.

    Returns:
        list: List of scalar values.
    '''
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

def interpolate_scalar_field(p_i, p_j, p_k, scalar_field):

    # Origin of new RF primed: O' = [0,0,0] but O = [p_i, p_j, p_k] in not primed RF.
    u, v, w = p_i - np.floor(p_i), p_j - np.floor(p_j), p_k - np.floor(p_k)

    # Find the eight coordinates enclosing the chosen point
    coordinates = find_enclosing_scalars(p_i, p_j, p_k)

    # Initialize interpolated_vector
    interpolated_scalar = 0.0
    cum_sum, cum_inv_dis= 0.0, 1.0

    if True:
      for coo in coordinates:
            # coordinates of cube vertices around point
            i, j, k = int(coo[0]), int(coo[1]), int(coo[2]) # O' position

            # distance from point to vertice of cube
            distance = magnitude([i - p_i, j - p_j, k - p_k]) # real distance
            #print(distance)

            # base case
            if distance <= 0.001:
                #print(f"known scalar field at [{p_i},{p_j},{p_k}]")
                return scalar_field[int(p_i)][int(p_j)][int(p_k)]

            cum_sum += scalar_field[i][j][k]/ (distance + 1e-10) ** 2
            #print(scalar_field[i][j][k],(distance + 1e-10) ** 2)
            cum_inv_dis += 1 / (distance + 1e-10) ** 2
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
            if distance <= 0.001:
                #print(f"known scalar field at [{p_i},{p_j},{p_k}]")
                return scalar_field[int(p_i)][int(p_j)][int(p_k)]

            cum_sum += scalar_field[i][j][k]*np.exp(-distance**(3)/8)/ (distance + 1e-10) ** 2
            cum_inv_dis += 1 / (distance + 1e-10) ** 2

    interpolated_scalar = cum_sum / cum_inv_dis
    return interpolated_scalar

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

global itera, scoord, posit, bmag

# Specify the file path
file_path = 'critical_points.txt'

# Displaying a message about reading from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line and create a list of dictionaries
data_list = [process_line(line) for line in lines[:] if process_line(line) is not None]

# Creating a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)

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

# Global Constants for Ionization Calculation

# Threshold parameters for the power-law distribution
global d, a, Lstar, Jstar, Estar

# Fraction of energy deposited locally (1 - d)
d = 0.82

# Exponent of the power-law distribution (a = 1 - d)
a = 1 - d

# Luminosity per unit volume for cosmic rays (eV cm^2)
Lstar = 1.4e-14

# Flux normalization constant (eV^-1 cm^-2 s^-1 sr^-1)
Jstar = 2.43e+15

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6

def PowerLaw(Eparam, E, power, const):
    """
    Power-law function to model cosmic ray distribution. (Energy Losses)

    Parameters:
    - Eparam (float): Reference energy scale.
    - E (float): Energy variable.
    - power (float): Exponent of the power-law.
    - const (float): Constant factor.

    Returns:
    - float: Computed value of the power-law function.
    """
    return const * (E / Eparam) ** (-power)

def ColumnDensity(sf, mu):
    """
    Compute column density for a given pitch angle and distance traveled.

    Parameters:
    - sf (float): Final distance traveled (stopping point of simulation).
    - mu (float): Cosine of the pitch angle (0 < pitch_angle < pi).

    Returns:
    - float: Computed column density.
    """

    dColumnDensity = 0.0
    index_sf = scoord.index(sf)  # Find index corresponding to the final distance
    Bats = bmag[index_sf]  # Magnetic field strength at the stopping point

    for i, sc in enumerate(scoord):
        if sc == sf:  # Stop simulation at the final distance
            break

        if i < 1:
            ds = scoord[1] - scoord[0]
        else:
            ds = scoord[i] - scoord[i-1]

        gaspos = index[i]  # Position for s in structured grid
        gasden = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density
        Bsprime = bmag[i]
        
        #print(gasden, gas_den[i][i][i])
        
        try:
            bdash = Bsprime / Bats  # Ratio of magnetic field strengths
            one_over = 1.0 / np.sqrt(1 - bdash * (1 - mu**2))  # Reciprocal of the square root term
        except ZeroDivisionError:
            print("Error: Division by zero. Check values of B(s')/B(s) and \mu")

        dColumnDensity += gasden * one_over  # Accumulate the contribution to column density

    print(dColumnDensity, gasden)

    return dColumnDensity, gasden

def Energy(E, mu, s, ds, cd, d=0.82): 
    """
    Compute new energy based on the given parameters.

    Parameters:
    - Ei (float): Initial energy.
    - mu (float): Cosine of the pitch angle (0 < pitch_angle < pi).
    - s (float): Distance traveled.
    - ds (float): Step size for distance.
    - cd (float): Column density up to s
    - d (float): Constant parameter (default value: 0.82).

    Returns:
    - float: New energy calculated based on the given parameters.
    """

    try:

        # Calculate the new energy using the given formula
        #new_energy = (E**(1 + d) - (1 + d) * Lstar * cd * Estar**(d))**(1 / (1 + d))

        Ei = (E**(1 + d) + (1 + d) * Lstar * cd * Estar**(d))**(1 / (1 + d))

    except Exception as e:
        # Catch potential issues with ColumnDensity() function
        print("Error:", e)
        print("ColumnDensity() has issues")
        exit()

    return Ei

def Jcurr(Ei, E, mu, s, ds, cd):
    """
    Calculate current J(E, mu, s) based on given parameters.

    Parameters:
    - Ei (float): Lower bound initial energy. E_exp
    - E (float): Integration variable.
    - mu (float): Pitch angle cosine.
    - s (float): Upper bound for distance integration (lower bound is 0.0).
    - ds (float): Distance between consecutive points [s, s + ds].

    Returns:
    - list: Three approximations of J(E, mu, s) based on different cases.
    """
    try:
        # Calculate energy using the Energy function
        pass
        #E = Energy(Ei, mu, s, ds, d)
    except Exception as e:
        # Catch potential issues with Energy() function
        print("Error:", e)
        print("Energy() has issues")
        exit()

    # Calculate Jcurr using the PowerLaw function
    Jcurr = PowerLaw(Estar, Ei, a, Jstar) * PowerLaw(Estar, Ei, d, Lstar) / PowerLaw(Estar, E, d, Lstar)

    # Approximations:
    # Case 1: E >> Estar => Ji(Ei) ~ J(E)
    Jcurr_one = PowerLaw(Estar, E, a, Jstar) #* PowerLaw(Estar, Ei, d, Lstar)

    # Case 2: E << Estar => Ji(Ei) ~ J(Estar)
    E__ = 1 + (1+d) * ((Lstar * (1 + d) * Estar * cd)** (1/ 1 + d))* E**d
    
    Jcurr_two = PowerLaw(Estar, E__, a, Jstar) * PowerLaw(Estar, Ei, d, Lstar)

    return Jcurr, Jcurr_one, Jcurr_two, cd

""" Ionization Calculation

- [x] Integrate over the trajectory to obtain column density
- [ ] Integrate over all posible energies E in \[1 MeV, 1GeV\]
- [ ] Integrate over all posible values of pitch angle d(cos(alpha_i)) with alpha_i in \[0, pi\]
- [ ] Add all three CR populations
"""

def Ionization(sf):
    # precision of simulation depends on data characteristics
    data_size = len(scoord) 
    #data_size = list(range(len(scoord)))
    
    # 1.60218e-6 ergs (1 MeV = 1.0 eV)
    Ei = 1.0e+6 # eV Ei
    Ef = 1.0e+9
    
    # ten thousand of precision to try
    dE  = (Ef-Ei)/data_size
    
    # 0 < mu < 1.0
    dmu = 1.0/data_size
    mu = [dmu*j for j in range(data_size)]

    # 0.0 < pitch < np.pi
    da = np.pi/data_size
    a = reversed([da*j for j in range(data_size)])
    
    dIo = 0.0
    print("Initial Conditions")
    #print(data_size, Ei, dE, da, dmu)
    testo = []

    for ai in a: 
        mui = np.cos(ai)
        #print(mui, ai, dIo)
        
            # Calculate column density using the provided function
        data = ColumnDensity(sf, mui)
        gasden = data[1]
        cd = data[0]
        

        dIo += dIo*np.sin(ai)*da # Ionization Differential

        E = Ei

        CurrentCaseZero  = []#J[0]#np.log10(J[0]) # Current using model
        CurrentCaseOne   = []#J[1]#np.log10(J[1]) # # Case 1: E >> Estar => Ji(Ei) ~ J(E)
        CurrentCaseTwo   = []#J[2]#np.log10(J[2]) # Case 2: E << Estar => Ji(Ei) ~ J(Estar)
        Energies = []#J[3]# np.log10(J[3]) # Coolumn Density at s

        for i, sc in enumerate(scoord[1:]):
            if sc > sf: # stop calculation at s final point
                break
            
            if i < 1:
                ds = scoord[1] - scoord[0]
            else:
                ds = scoord[i] - scoord[i-1]

            # E in 1 MeV => 1 GeV
            E = Ei + i*dE

            # E_exp = Ei^(1+d) = E^(1+d) + L_(1+d) N E_^d   
            E_exp = Energy(E, mui, sc, ds, cd, d) 

            # Current for J_+(E, mu, s)
            J = Jcurr(E_exp, E, mui, sc, ds, cd)
            
            CurrentCaseZero.append(np.log10(J[0])) # Current using model
            CurrentCaseOne.append(np.log10(J[1]))  # Case 1: E >> Estar => Ji(Ei) ~ J(E)
            CurrentCaseTwo.append(np.log10(J[2]))  # Case 2: E << Estar => Ji(Ei) ~ J(Estar)
            Energies.append(np.log10(E))           # Coolumn Density at s
                   
            #print(f"{i} Calculated value: {J[0]}, {J[1]},{J[0]},{E}, {cd}, {gasden} ")
            try:
                dIo += 0.0# J*dE*np.sin(ai)*da
            except Exception as e:
                print(e)
                print("Jcurr() has issues")

        break

    return (CurrentCaseZero , CurrentCaseOne, CurrentCaseTwo ,  Energies) #dIo

# Choose a test case for the streamline coordinate
sf = scoord[1187]  # test case
sf = scoord[-1]  # test case

# Uncomment the line below if you want to test a different streamline coordinate
sf = scoord[1186]  # test case

# Test Ionization function and print the result
ionization_result = Ionization(sf)

CurrentCaseZero = ionization_result[0] # Current using model
CurrentCaseOne  = ionization_result[1] # # Case 1: E >> Estar => Ji(Ei) ~ J(E)
CurrentCaseTwo  = ionization_result[2] # Case 2: E << Estar => Ji(Ei) ~ J(Estar)
Energies  = ionization_result[3] # Coolumn Density at s

logscoord = [np.log10(s) for s in scoord[1:]]

print(len(logscoord), len(CurrentCaseZero), len(CurrentCaseOne), len(CurrentCaseTwo))
import matplotlib.pyplot as plt

# Create a 1x3 subplot grid
fig, axs = plt.subplots(3, 1, figsize=(8, 15))

# Scatter plot for Case Zero
axs[0].scatter(Energies, CurrentCaseZero, label='$J(\mu, s, E)$', marker='x', color='blue')

# Scatter plot for Case One
axs[1].scatter(Energies, CurrentCaseOne, label='$E >> E_{**} \| J_i(E_i) ~ J_i(E)$', marker='*', color='green')

# Scatter plot for Case Two
axs[2].scatter(Energies, CurrentCaseTwo, label='$E << E_{**} \| J_i(E_i) ~ J(E_*)$', marker='+', color='red')
axs[2].set_xlabel('$Log_{10}(E \ eV)$')

# Set y-axis labels for all subplots
for ax in axs:
    ax.set_ylabel('$Log_{10}(J eV^{-1} cm^{-2} s^{-1} sr^{-1})$')

# Add legends to each subplot
axs[0].legend()
axs[1].legend()
axs[2].legend()

# Adjust layout for better spacing
#plt.tight_layout()

# Display the plot
plt.show()

# Characterization of the path - Find peaks in the trajectory
x = bmag.copy()[1:]

# Find peaks using the find_peaks function
peaks, _ = find_peaks(x, height=0)
maxes = [x[index] for index in peaks]
bmag_maxes = [bmag[peak] for peak in peaks]

# Print the indices of the peaks
print(peaks)

# Plot the trajectory with peaks marked
plt.plot(x)
plt.hlines(maxes, xmin=0, xmax=len(scoord), color="grey", linestyle="--")
plt.plot(peaks, [x[i] for i in peaks], "x")
plt.show()

#print(sf)