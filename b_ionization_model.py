# import for visualization
import matplotlib.pyplot as plt
from z_library import *
import numpy as np
import random
import sys


"""  
Methods
"""

gas_den = np.array(np.load("input_data/gas_number_density.npy", mmap_mode='r'))

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

# Specify the file path
#file_path = 'critical_points.txt'

if len(sys.argv) < 1: # if trajectory has been saved up in a file
    file_path = sys.argv[1]

    # Displaying a message about reading from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process each line and create a list of dictionaries
    data_list = [process_line(line) for line in lines[:] if process_line(line) is not None]

    # Creating a DataFrame from the list of dictionaries
    #df = pd.DataFrame(data_list)

    # Extracting data into separate lists for further analysis
    itera, distance, posit, xpos, ypos, zpos, field_v, bfield, field_x, field_y, field_z, index = [], [], [], [], [], [], [], [], [], [], [], []

    for iter in data_list: # Data into variables
        itera.append(iter['iteration'])
        distance.append(iter['trajectory (s)'])
        posit.append(iter['Initial Position (r0)'])
        xpos.append(iter['Initial Position (r0)'][0])
        ypos.append(iter['Initial Position (r0)'][1])
        zpos.append(iter['Initial Position (r0)'][2])
        field_v.append(iter['field vector'])
        bfield.append(iter['field magnitude'])
        field_x.append(iter['field vector'][0])
        field_y.append(iter['field vector'][1])
        field_z.append(iter['field vector'][2])
        index.append(iter['indexes'])
    print(" Data Successfully Loaded")
else:
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
    TOTAL_TIME = 900000
    TIMESTEP   = 0.05
    SNAPSHOTS  = int(TOTAL_TIME/TIMESTEP)
    DS         = 128
    MARGIN     = 34

    # this points contains several pocket, and it vanishes at the extremes.

    point_i = 47.657
    point_j = 81.482
    point_k = 35.057

    if False: # random point generator
        point_i = random.uniform(MARGIN, DS-MARGIN)
        point_j = random.uniform(MARGIN, DS-MARGIN)
        point_k = random.uniform(MARGIN, DS-MARGIN)

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
        for time in timestep:
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
            print(magnitude(cur_pos,  prev_pos) * scale_factor)
            prev_pos =  cur_pos.copy()

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

    """ 
    plt.plot(distance, bfield, linewidth=2.0)
    plt.show() 
     """    

if True:
    with open(f"c_field_lines.txt", "w") as c1_data:
        for count in range(len(distance)):
            c1_data.write(f"{count}, {distance[count]},{bfield[count]}\n") 

# Global Constants for Ionization Calculation

# Threshold parameters for the power-law distribution
global d, a, Lstar, Jstar, Estar, epsilon

# mean energy epsilon lost by a CR particle per ionization event
epsilon = 0.028837732137317718 #eV # 2.1 #

# Fraction of energy deposited locally (1 - d)
d = 0.82

# Exponent of the power-law distribution
a = 0.1

# Luminosity per unit volume for cosmic rays (eV cm^2)
Lstar = 1.4e-14

# Flux constant (eV^-1 cm^-2 s^-1 sr^-1)
C = 2.43e+15            # Proton in Low Regime (A. Ivlev 2015) https://iopscience.iop.org/article/10.1088/0004-637X/812/2/135
Enot = 500e+6
Jstar = 2.4e+15*(10e+6)**(0.1)/(Enot**2.8)

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
    return const * (E / Eparam) ** (power)

def ColumnDensity(sh, mu):
    """
    Compute column density for a given pitch angle and distance traveled.

    Parameters:
    - sh (float): Final distance traveled (reachpoint of column density).
    - mu  (float): Cosine of the pitch angle (0 < pitch_angle < pi).

    Returns:
    - float: Computed column density.
    """

    dColumnDensity = 0.0
    index_sh = distance.index(sh)  # final distance will be highetst # Find index corresponding to the final distance
    Bats = bfield[index_sh]  # Magnetic field strength at the stopping point
    prev_sc = distance[0]
    ds = abs(distance[1] - distance[0]) # they are equally space (unito * delta)
    protonMass = 1.6726219259e-24

    for i, sc in enumerate(distance):

        gaspos = radius_vector[i]  # Position for s in structured grid
        gasden = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density order of 1.0^0
        numb_den = gasden+0*(1/protonMass)
        Bsprime = bfield[i]
        
        #print(f"gas density at: {sc} =>",gasden, " w/ N =>", numb_den)
        
        try:
            bdash = Bsprime / Bats  # Ratio of magnetic field strengths
            deno = 1 - bdash * (1 - mu**2)
            if deno < 0:
                return dColumnDensity
            one_over = 1.0 / np.sqrt(deno)            # Reciprocal of the square root term
            dColumnDensity += numb_den * ds * one_over   # Accumulate the contribution to column density
        except ZeroDivisionError:
            print("Error: Division by zero. Check values of B(s')/B(s) and \mu")
            return dColumnDensity
        
        return dColumnDensity

def Energy(E, mu, cd, d=0.82): 
    """
    Compute new energy based on the given parameters.

    Parameters:
    - Ei (float): Initial energy.
    - mu (float): Cosine of the pitch angle (0 < pitch_angle < pi).
    - s (float): Distance traveled.
    - cd (float): Column density up to s
    - d (float): Constant parameter (default value: 0.82).

    Returns:
    - float: New energy calculated based on the given parameters.
    """
    try:
        # Calculate the new energy using the given formula
        Ei = (E**(1 + d) + (1 + d) * Lstar * cd * Estar**(d))**(1 / (1 + d))

    except Exception as e:
        # Catch forbiden values in Ei expression
        print("Error:", e)
        exit()

    return Ei

def Jcurr(Ei, E, cd):
    """
    Calculate current J(E, mu, s) based on given parameters.

    Parameters:
    - Ei (float): Lower bound initial energy. E_exp
    - E (float): Integration variable.
    - mu (float): Pitch angle cosine.

    Returns:
    - list: J(E, mu, s) 
    """
    try:
        # Calculate Jcurr using the PowerLaw function
        Jcurr = PowerLaw(Estar, Ei, a, Jstar) * PowerLaw(Estar, Ei, -d, Lstar) / PowerLaw(Estar, E, -d, Lstar)
        
        JcurrIvlev = C*(E**0.1)/((E+500)**2.8) #PowerLaw(Estar, Ei, a, Jstar) * PowerLaw(Estar, Ei, -d, Lstar)

    except Exception as e:
        print(PowerLaw(Estar, Ei, a, Jstar), '+')
        print(PowerLaw(Estar, Ei, -d, Lstar))
        print(Estar, E, -d, Lstar)
        print("Error:", e)
        print("Jcurr() has issues")
        exit()

    return Jcurr, JcurrIvlev

""" 
Ionization Calculation

- [x] Integrate over the trajectory to obtain column density
- [x] Integrate over all posible energies E in \[1 MeV, 1GeV\]
- [x] Integrate over all posible values of pitch angle d(cos(alpha_i)) with alpha_i in \[0, pi\]
- [ ] Add all three CR populations
"""

def Ionization(reverse, mirror=False):
    with open(f"b_output_data/ionization_data.txt", "w") as io_data: # save data for future analysis

        # precision of simulation depends on data characteristics (around precision 2500)
        precision = 1000

        pocket, global_info = visualize_pockets(bfield, 0, plot=False) # this plots
        index_pocket, field_pocket = pocket[0], pocket[1]
        print(index_pocket)

        globalmax_index = global_info[0]
        globalmax_field = global_info[1]

        # in the case of mirroring we'll have $\mu_i < \mu <\mu_{i+1}$ between the ith-pocket 
        def calculate_mu(B_i, B_j):
            # B_j > B_i
            return np.sqrt((1 - B_i / B_j) )        

        # 0.0 < pitch < np.pi/2
        da = np.pi / (precision)
        ang = np.array([ da * j for j in range(int(precision)) ])

        match mirror:
            case True: # se up a list of \mu's for each pocket
                da = np.pi / (precision)
                ang = np.array([ da * j for j in range(int(precision)) ])
                #mu = np.cos(ang)  
            case False: 
                da = np.pi / (precision)
                ang = np.array([ da * j for j in range(int(precision)) ])
                #mu = np.cos(ang)
            
        # Forward moving particles (-1 < \mu < \mu_h) where \mu_h is at the lowest peak 
        ionization_pop = 0.0
        
        # 1.60218e-6 ergs (1 MeV = 1.0e+6 eV)
        Ei = 1.0e+3 # eV
        Ef = 1.0e+9
        
        # ten thousand of precision to try
        dE = (Ef - Ei) / precision
        E = [Ei + k*dE for k in range(precision)]
        logE = [np.log10(Eval) for Eval in E]

        #print("Initial Conditions")
        #print("Precision (No. of loops): ", precision, "Ei: ", Ei, "dE (eV): ", dE, "da: ", da,"\n")
        
        ColumnH2       = []
        LogIonization  = []

        for ai in reversed(ang):   

            mui = np.cos(ai)
            
            cd = 3.577442927981429e+19 #ColumnDensity(distance[globalmax_index], mui) # 0.000, np.inf #

            if np.isnan(cd): # tests if cd = Nan
                print("cd is Nan? ", np.isnan(cd)==True)
                continue
            
            Spectrum       = []
            Spectrumi      = []
            IvlevLaw       = []
            Evar = Ei

            for k in range(precision):
                
                Evar = Ei + k * dE

                # E_exp = Ei^(1+d) = E^(1+d) + L_(1+d) N E_^d   
                E_exp = Energy(Evar, mui, cd, d) # Ei(E)

                # Current for J_+(E, mu, s) # Jcurr(Ei, E, cd)
                J, _ = Jcurr(E_exp, Evar, cd)
                J_i  = PowerLaw(Estar, E_exp, a, Jstar)*PowerLaw(Estar, E_exp, -d, Lstar) / PowerLaw(Estar, Evar, -d, Lstar)
                print(J, _,J_i)
                
                    # Current using model
                Spectrum.append(np.log10(J))
                Spectrumi.append(np.log10(J_i))
                IvlevLaw.append(np.log10(_))
                
                #print("Ionization (s): ", ionization_pop, "related to energy (J):", Evar, "with angle(alpha):", ai) 
                
                try:
                    print(1/epsilon, J, dE, np.sin(ai), da)
                    #ionization_pop += (1/epsilon)*J*dE*np.sin(ai)*da
                except Exception as e:
                    print(e)
                    print("J Spectrum() has issues")
                io_data.write(f"{np.log10(ionization_pop)}, {cd}\n") 

            ColumnH2.append(cd)
            LogIonization.append(np.log10(ionization_pop)) # Ionization for that Column Density for that population
        
        print("Resulting Ionization: ", ionization_pop)       

    return (LogIonization, ColumnH2, logE, Spectrum, Spectrumi, IvlevLaw) 

# Choose a test case for the streamline coordinate

# Calculating different Populations

# Forward moving particles (-1 < \mu < \mu_l) where \mu_h is at the lowest peak $\mu_l = \sqrt{1-B(s)/B_l}$
forward_ionization = Ionization(reverse = False)

# Backward moving particles (-1 < \mu < \mu_h) where \mu_h is at the highest peak $\mu_h = \sqrt{1-B(s)/B_h}$
#backward_ionization = Ionization(reverse = True)

# such that s_h and s_l form a pocket

# Mirrored particles (Union of consecutive \mu_l < \mu < \mu_h given by pockets)
# mirrored_ionization = Ionization(reverse=False, mirror = True)

logIonization = forward_ionization[0] # Current using model
ColumnH       = forward_ionization[1] # 
LogEnergies   = forward_ionization[2] # 
Spectrum      = forward_ionization[3] # 
Spectrumi     = forward_ionization[4] # 
IvlevLaw      = forward_ionization[5] # 

# Save data in files
precision = len(logIonization)
print(len(ColumnH))
print(len(Spectrum))    # +1

da = np.pi / 2*(precision)
ang = np.array([ da * j for j in range(int(precision)) ])

if True:
    # Create a 2x1 subplot grid
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))

    # Scatter plot for Spectrum vs LogEnergies
    
    axs[0].plot(LogEnergies, Spectrum, label='log_{10}(J(E) \ eV^-1 cm^-2 s^-1 sr^-1)', linestyle=':', color='red')
    axs[0].plot(LogEnergies,Spectrumi, label='log_{10}(J_i(E_i) \ eV^-1 cm^-2 s^-1 sr^-1)', linestyle=':', color='blue')
    axs[0].plot(LogEnergies, IvlevLaw, label='log_{10}(J(E) (Ivlev et al, 2015))', linestyle=':', color='grey')
    #axs[1].set_yscale('log')
    axs[0].set_ylabel('log_{10}(J(E) \ eV^-1 cm^-2 s^-1 sr^-1)')
    axs[0].set_xlabel('Log_{10}(E \ eV)')
    axs[0].legend()


    # Scatter plot for Ionization vs Column H
    #axs[0].plot(ColumnH, logIonization, label='$log_{10}(\zeta \ Forward Particles)$', linestyle='--', color='blue')
    axs[1].plot(ColumnH, logIonization, label='$log_{10}(\zeta \ Forward Particles)$', linestyle='--', color='blue')
    axs[1].set_ylabel('$log_{10}(\zeta \ s^-1)$')
    axs[1].set_xlabel('$n(H_2) (N \ cm^-2)$')
    axs[1].legend()

    # Adjust layout
    #plt.tight_layout()

    # Save Figure
    plt.savefig("Ion_ColumnD_and_Spectrum.png")

    # Display the plot
    #plt.show()