# import for visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from z_library import *
import numpy as np
import random

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

if True:
    with open(f"c_field_lines.txt", "w") as c1_data:
        for count in range(len(distance)):
            c1_data.write(f"{count}, {distance[count]},{bfield[count]}\n") 


"""  
Finished Field Lines Integration: loaded data into c_field_lines.txt

"""

"""  
Reproduce M. Padovani et al.: Cosmic-ray ionisation in circumstellar discs Ionization.

"""
size = 200
precision = size
Ei = 1.0e+0
Ef = 1.0e+15

N0 = 10e+19
Nf = 10e+27

n0 = 150 #cm−3 and 
k  = 0.5 # –0.7

d = 0.82
a = 0.1 # spectral index either 0.1 from Low Model, or \in [0.5, 2.0] according to free streaming analytical solution.

# mean energy lost per ionization event
epsilon = 35.14620437477293

# Luminosity per unit volume for cosmic rays (eV cm^2)
Lstar = 1.4e-14

# Flux constant (eV^-1 cm^-2 s^-1 sr^-1) C*(10e+6)**(0.1)/(Enot+6**2.8)
Jstar = 2.43e+15*(10e+6)**(0.1)/(500e+6**2.8) # Proton in Low Regime (A. Ivlev 2015) https://iopscience.iop.org/article/10.1088/0004-637X/812/2/135

# Flux constant (eV^-1 cm^-2 s^-1 sr^-1)
C = 2.43e+15            # Proton in Low Regime (A. Ivlev 2015) https://iopscience.iop.org/article/10.1088/0004-637X/812/2/135
Enot = 500e+6
Jstar = 2.4e+15*(1.0e+6)**(0.1)/(Enot**2.8)

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6
Epion = 280e+6

""" 
Column Densities

    - sim_column_density: list(list()) => [for mu = 0/size,[N0, N1, ..., Nf], for mu=1/size, ...]
    - indie_column_density: list() => [10^3 => 10^27] as done in b1_padovani.py
"""

muforward   = np.array([k/(50) for k in range(50)]) # Backward Ionization
mubackward  = np.array([k/(50)-1 for k in range(50)]) # Backward Ionization

energy      = np.array([1.0e+2*(10)**(14*k/size) for k in range(size)])  # Energy values from 1 to 10^15
diff_energy = np.array([(10)**(14*k/size) for k in range(size)])
logenergy   = np.log10(energy)                                        # log10(Energy) values from 0 to 15
column_density = np.array([1.0e+19*(10)**(8*k/precision) for k in range(precision)])

print("Extremes of Independent N ", np.log10(column_density[0]), np.log10(column_density[-1]))

column_density_forward  = {}
column_density_backward = {}
#column_density_forward  = []
#column_density_backward = []

dmu = 1/precision

############----------------------------------FORWARD----------------------------------############

for mui in muforward:
    column_density_forward[mui] = 0.0
    Bats                    = bfield[-1]                     # Magnetic field strength at the stopping point
    dcolumn_density         = 0.0
    ds                      = abs(distance[1] - distance[0]) # they are equally space (unito * delta)
    for i in range(len(distance)): # this        
        gaspos    = radius_vector[i]  # Position for s in structured grid
        numb_den  = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density order of 1.0^0       
        cur_distance = distance[i]
        Bsprime   = bfield[i]
        bdash     = Bsprime / Bats  # Ratio of magnetic field strengths
        deno      = 1 - bdash * (1 - mui**2)
        
        if deno < 0:
            column_density_forward[mui] = dcolumn_density
            break

        one_over         = 1.0 / np.sqrt(deno)               # Reciprocal of the square root term
        dcolumn_density += numb_den * ds * one_over   # Accumulate the contribution to column density
    column_density_forward[mui] = dcolumn_density
    #column_density_forward.append(dcolumn_density)

############----------------------------------BACKWARD----------------------------------############

for mui in mubackward:
    column_density_backward[mui] = 0.0
    Bats                    = bfield[0]                     # Magnetic field strength at the stopping point
    dcolumn_density         = 0.0
    ds                      = abs(distance[1] - distance[0]) # they are equally space (unito * delta)
    for i in range(len(distance)-1, 0, -1): # this     
        gaspos    = radius_vector[i]  # Position for s in structured grid
        numb_den  = interpolate_scalar_field(gaspos[0], gaspos[1], gaspos[2], gas_den)  # Interpolated gas density order of 1.0^0       
        cur_distance = distance[i]
        Bsprime   = bfield[i]
        bdash     = Bsprime / Bats  # Ratio of magnetic field strengths
        deno      = 1 - bdash * (1 - mui**2)
        
        if deno < 0:
            column_density_backward[mui] = dcolumn_density
            break
        one_over         = 1.0 / np.sqrt(deno)               # Reciprocal of the square root term
        dcolumn_density += numb_den * ds * one_over   # Accumulate the contribution to column density
    column_density_backward[mui] = [dcolumn_density]
    #column_density_backward.append(dcolumn_density)

if True:

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create a 2x1 grid of subplots

    #colors = cm.rainbow(np.linspace(0, 1, len(list(column_density_forward.keys())[0:])))

    count = 0
    #for c, mui in zip(colors, list( sim_column_density_forward.keys())[0:] ):
    #    axs[0].plot(sim_column_density_forward[mui], label=f'N($\mu={mui}$)', linestyle=":", color=c)
    #    count += 1

    axs[0].plot(muforward, column_density_forward.values(), label=f'N($\mu$)', linestyle=":", color="black")
    axs[0].set_yscale('log')
    axs[0].set_title('Forward Column Densities')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Column Density $log_10(N(\mu))$ (log scale)')
    axs[0].grid(True)

    #axs[0].legend()
    #colors = cm.rainbow(np.linspace(0, 1, len(list(column_density_backward.keys())[0:])))

    count = 0
    #for c, mui in zip(colors, list(sim_column_density_backward.keys())[0:]):
    #    axs[1].plot(sim_column_density_backward[mui], label=f'N($\mu={mui}$)', linestyle=":", color=c)
    #    count += 1
    axs[1].plot(mubackward, column_density_backward.values(), label=f'N($\mu$)', linestyle=":", color="black")
    axs[1].set_yscale('log')
    axs[1].set_title('Backward Column Densities')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Column Density $log_10(N(\mu))$ (log scale)')
    axs[1].grid(True)
    #axs[1].legend()


    plt.tight_layout(pad=5)
    plt.savefig("b_output_data/b_sim_column_densities_mosaic.png")
    plt.show()

""" 
Ionization Calculation

- [x] Integrate over the trajectory to obtain column density
- [ ] Integrate over all posible energies E in \[1 MeV, 1 GeV\]
- [ ] Integrate over all posible values of pitch angle d(cos(alpha_i)) with alpha_i in \[0, pi/2\]
- [ ] Add all three CR populations

"""
energy = np.array([1.0e+2*(10)**(14*k/size) for k in range(size)])  # Energy values from 1 to 10^15
diff_energy = np.array([(10)**(14*k/size) for k in range(size)])
logenergy = np.log10(energy)                                        # log10(Energy) values from 0 to 15
column_density = np.array([1.0e+19*(10)**(8*k/size) for k in range(size)])

# for one value of \mu only
log_lossE = []
log_lossEi = []
proton_local_spectrum = []
log_proton_ism_spectrum = []
low_energy_proton_ism_spectrum = []
forward_spectrum = []
integrand_jl = {}
proton_spectrum = 0.0
ism_spectrum = 0.0
zeta_Ni  = 0.0

log_proton_spectrum = lambda E: np.log10(C) + 0.1*np.log10(E) - 2.8*np.log10(Enot + E) # J_p as in (Ivlev, 2015), Padovani et al equation (1)
proton_spectrum = lambda E: C*E**0.1/(Enot + E)**2.8
log_ism_spectrum = lambda x: np.log10(Jstar) + a*(np.log10(x) - np.log10(Estar))
log_loss_function = lambda z: np.log10(Lstar) - d*(np.log10(z) - np.log10(Estar) )

ism_spectrum = lambda x: Jstar*(x/Estar)**a
loss_function = lambda z: Lstar*(Estar/z)**d

log_ion = []

for i, Ni in enumerate(column_density): # will go from 3,4,5,6--- almost in integers#

    jl_dE = 0.0

    for k, E in enumerate(energy): # will go from 3,4,5,6--- almost in integers#

        Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Ni)**(1 / (1 + d)) # E_i(E, N)

        isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
        llei = loss_function(Ei)           # log_10(L(E_i))
        jl_dE += isms*llei*diff_energy[k] # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
    zeta_Ni = jl_dE/epsilon/epsilon #jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon
    log_ion.append(np.log10(zeta_Ni))

print("Extremes of zeta(N) Padovani et al: ", log_ion[0], log_ion[-1])

integrand_forward_jl = {}
forward_logz = []

""" 
# forward moving Cosmic Rays 

"""

#for i, Ni in enumerate(column_density_forward): # will go from 3,4,5,6--- almost in integers#
log_forward_ion = []

for i, Ni in enumerate(column_density): # will go from 3,4,5,6--- almost in integers#

    jl_dE = 0.0

    for k, E in enumerate(energy): # will go from 3,4,5,6--- almost in integers#

        Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Ni)**(1 / (1 + d)) # E_i(E, N)

        isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
        llei = loss_function(Ei)           # log_10(L(E_i))
        jl_dE += isms*llei*diff_energy[k] # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
    zeta_Ni = jl_dE/epsilon #jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon
    
    log_forward_ion.append(np.log10(zeta_Ni))

print("Extremes of zeta(N) forward: ", log_forward_ion[0], log_forward_ion[-1])

""" 
# backward moving Cosmic Rays 

"""
log_backward_ion = []

#for i, Ni in enumerate(column_density_backward):
for i, Ni in enumerate(column_density): # will go from 3,4,5,6--- almost in integers#

    # integrate with mui here <==

    jl_dE = 0.0

    for k, E in enumerate(energy): 

        Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* 0.0)**(1 / (1 + d)) # E_i(E, N)
 
        isms  = ism_spectrum(Ei)            # log_10(j_i(E_i))
        llei  = loss_function(Ei)           # log_10(L(E_i))
        jl_dE += isms*llei*diff_energy[k]   # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
    zeta_Ni = jl_dE/epsilon #jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon
    
    log_backward_ion.append(np.log10(zeta_Ni))

print("Extremes of zeta(N) backward", log_backward_ion[0], log_backward_ion[-1])

"""  
Mirrored Particles

"""
(indexes, peaks), (index_global_max, upline) = pocket_finder(bfield, 0, plot=False)

# lets make a list of pairs corresponding to regions considered pockets

zeta_Ni = 0.0
mirrored_logz = []
integrand_mirrored_jl = {}









"""  
\mathcal{L} Model: Protons

"""
model_L = [-3.331056497233e+6, 1.207744586503e+6,-1.913914106234e+5,
            1.731822350618e+4,-9.790557206178e+2, 3.543830893824e+1, 
           -8.034869454520e-1, 1.048808593086e-2,-6.188760100997e-5, 
            3.122820990797e-8]

logz = []

differential_fit = lambda z2,z1,e2,e1: (z2-z1)/(e2-e1) # slope of a log function log(F_2) = m log(e2) + const.
derivative_fit = []

for i,Ni in enumerate(column_density):
    lz = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
    logz.append(lz)

    difflogfit = (0.0)

print("Extremes of fitting zeta(N) ",logz[0], logz[-1])

logzetafit = np.array(logz)
svnteen = np.ones_like(column_density)*(-17)

"""  
PLotting 

"""
del a

s_of_m = ["#F4A6D7", "#D10056", "#7D2248"]

""" 
Free Streaming Cosmic Rays, Analytical Expression (Silsbee & Ivlev, 2019)

"""
fig, axs = plt.subplots(2, 2, figsize=(10,10))  # Create a 2x2 grid of subplots
If = {
    "a=0.4": 5.14672,
    "a=0.5": 3.71169,
    "a=0.7": 2.48371,
    "a=0.9": 1.92685,
    "a=1.1": 1.60495
    }

Nof = Estar/((1+d)*Lstar)

# Colors for plotting
colors = cm.rainbow(np.linspace(0, 1, len(If)))

# Plot each curve and add text annotations
for c, (b, I) in zip(colors, If.items()):
    free_streaming_ion = []
    a = float(b.split("=")[1])
    gammaf = (a + d - 1) / (1 + d)
    
    for Ni in column_density:
        fs_ionization = (1 / epsilon) * (1 + d) / (a + 2 * d) * Jstar * Lstar * Estar * I * (Ni / Nof) ** (-gammaf)
        free_streaming_ion.append(np.log10(fs_ionization))
    
    axs[0, 0].plot(column_density, free_streaming_ion, label=f'$log(\zeta_f(N, a={a}))$', linestyle="--", color=c)
    axs[0, 0].text(column_density[-1], free_streaming_ion[-1], f'a={a}, $\gamma_f$={gammaf:.2f}', fontsize=10, color=c)

# Plot Ionization vs Column Density (Second plot)
axs[0,0].plot(column_density, logzetafit, label='$log_{10}(\zeta) \, (\mathrm{Padovani \, et \, al})$', linestyle="--", color="grey")
axs[0,0].plot(column_density, log_ion, label='$log_{10}(\zeta_{\int dE})$', linestyle="-", color="black")
axs[0,0].plot(column_density, svnteen, label='$\zeta = 10^{-17}$', linestyle=":", color="dimgrey")
axs[0,0].set_xscale('log')
axs[0,0].set_title('Ionization vs Column Density')
axs[0,0].set_xlabel('Column Density (log scale)')
axs[0,0].set_ylabel('Ionization (log scale)')
axs[0,0].legend()
axs[0,0].grid(True)

# Plot Spectrum j_p vs Energy (Third plot) 
axs[0,1].set_xscale('log')
axs[0,1].set_title('Spectrum $j_p$ vs Energy')
axs[0,1].set_xlabel('Energy ($log_{10}(E / eV)$)')
axs[0,1].set_ylabel('Spectrum $j_p$')
#axs[0,1].set_ylim([-180, 0])
#axs[0,1].legend()
axs[0,1].grid(True)


# Plot Ionization vs Column Density (Second plot)
#axs[1,0].plot(column_density_forward.values(), forward_logzeta, label='$log_{10}(\zeta_f) \, (Forward \ CRs)$', linestyle="--", color="blue")
#axs[1,0].scatter(column_density_forward.values(), forward_logzeta, marker="*", color="blue")
#axs[1,0].plot(column_density_backward.values(), backward_logzeta, label='$log_{10}(\zeta_b) \, (Backwards \ CRs)$', linestyle="--", color="red")
#axs[1,0].scatter(column_density_backward.values(), backward_logzeta, marker="|", color="red")
axs[1,0].plot(column_density, svnteen, label='$\zeta = 10^{-17}$', linestyle="--", color="black")
axs[1,0].plot(column_density, log_forward_ion, label='$\zeta = 10^{-17}$', linestyle="--", color="black")
axs[1,0].set_xscale('log')
axs[1,0].set_title('Ionization Populations Using Margo\'s Data')
axs[1,0].set_xlabel('Column Densities (log scale)')
axs[1,0].set_ylabel('Ionization (log scale)')
axs[1,0].legend()
axs[1,0].grid(True)

# Plot Spectrum L(E) vs Energy (Fourth plot)
axs[1,1].set_title('Loss Function vs Energy')
axs[1,1].set_xlabel('Energy ($log_{10}(E / eV)$)')
axs[1,1].set_ylabel('Spectrum $j_p$')
#axs[1,1].legend()
axs[1,1].grid(True)

fig.tight_layout(pad=2.0)
plt.savefig("b_output_data/b_ionizations.png")

#plt.show()
