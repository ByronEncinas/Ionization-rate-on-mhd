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

distance      = np.array(list(left_distance) + [f + d for d in right_distance[1:]])
radius_vector = np.array(list(left_radius_vector) + right_radius_vector[1:])
bfield        = np.array(list(left_bfield_magnitudes) + right_bfield_magnitudes[1:])

if True:
    with open(f"c_field_lines.txt", "w") as c1_data:
        for count in range(len(distance)):
            c1_data.write(f"{count}, {distance[count]},{bfield[count]}\n") 


"""  
Finished Field Lines Integration: saved data into c_field_lines.txt

"""

"""  
Reproduce M. Padovani et al.: Cosmic-ray ionisation in circumstellar discs Ionization.

"""
size = 200

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

""" 
Ionization Calculation

- [x] Integrate over the trajectory to obtain column density
- [x] Integrate over all posible energies E in \[1 MeV, 1 GeV\]
- [x] Integrate over all posible values of pitch angle cosine
- [ ] Add all three CR populations

"""
energy = np.array([1.0e+2*(10)**(14*k/size) for k in range(size)])  # Energy values from 1 to 10^15
diff_energy = np.array([energy[k]-energy[k-1] for k in range(len(energy))])
diff_energy[0] = energy[0]
column_density = np.array([1.0e+19*(10)**(8*k/size) for k in range(size)])

print("Extremes of Independent N ", np.log10(column_density[0]), np.log10(column_density[-1]))

""" 
# Column Density as Independent Variable

Using Numpy Arrays

"""

# for one value of \mu only
zeta_N  = 0.0

ism_spectrum = lambda x: Jstar*(x/Estar)**a
loss_function = lambda z: Lstar*(Estar/z)**d

log_ion = np.zeros((len(column_density)))

for j, Nj in enumerate(column_density): # will go from 3,4,5,6--- almost in integers#

    jl_dE = 0.0

    for k, E in enumerate(energy): # will go from 3,4,5,6--- almost in integers#

        Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) # E_i(E, N)

        isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
        llei = loss_function(Ei)           # log_10(L(E_i))
        jl_dE += isms*llei*diff_energy[k]  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
        Jacobian = 1.0

    zeta_N = Jacobian * jl_dE / epsilon                 # \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon

    log_ion[j] = np.log10(zeta_N)

print("Extremes of zeta(N) Padovani et al: ", log_ion[0], log_ion[-1])

""" 
# Import Column Densities for Forward and Backward Moving Particles

"""
mu_ism    = np.array(np.load("PitchAngleCosines.npy", mmap_mode='r'))
diff_mu_ism = np.array([mu_ism[i-1]-mu_ism[i] for i in range(len(mu_ism))]) #mu_ism is decreasing
diff_mu_ism[0] = mu_ism[0]
mu_ism_x_diff = mu_ism*diff_mu_ism

ForwardColumn  = np.array(np.load("ForwardColumn.npy", mmap_mode='r'))
BackwardColumn = np.array(np.load("BackwardColumn.npy", mmap_mode='r'))

if False:
    import matplotlib.cm as cm

    colors = cm.rainbow(np.linspace(0, 1, len(mu_ism)))

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create a 2x1 grid of subplots

    for i, c in enumerate(colors):
        axs[0].scatter(distance/10e+18, ForwardColumn[i,:], label=f'$N_+$-map', linestyle="--", color=c, s =5)

    axs[0].set_yscale('log')
    axs[0].set_ylabel('$log_10(N_+(\mu))$ (log scale)')
    #axs[0].legend()
    axs[0].grid(True)
    
    for i, c in enumerate(colors):
        axs[1].scatter(distance/10e+18, BackwardColumn[i,:], label=f'$N_-$-map', linestyle="--", color=c, s=5)

    axs[1].set_yscale('log')
    axs[1].set_xlabel('Distance ($s$ $cm/10^{18}$)')
    axs[1].set_ylabel('$log_10(N_-(\mu_i))$ (log scale)')
    #axs[0].legend()
    axs[1].grid(True)
    
    plt.show()

""" 
# forward moving Cosmic Rays 

"""

c = (len(mu_ism), len(ForwardColumn[0,:])) #ForwardColumn[0,:] = ForwardColumn[mu=1,:] so its expected is the biggest 1D of all
log_forward_ion = np.zeros(c)

for i, mui in enumerate(mu_ism):

    for j, Nj in enumerate(ForwardColumn[i,:]): 
    
        jl_dE = 0.0
    
        for k, E in enumerate(energy): # will go from 3,4,5,6--- almost in integers#

            Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) # E_i(E, N)

            isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
            llei = loss_function(Ei)           # log_10(L(E_i))
            jl_dE += isms*llei*diff_energy[k] # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
        """
             if 1 - (bfield[j]/bfield[0])*(1 - mui**2) > 0:
                mu_local = np.sqrt(1 - (bfield[j]/bfield[0])*(1 - mui**2))
                Jacobian = (mui/mu_local)*(bfield[j]/bfield[0])    
            else:
                Jacobian = 0.0
                break
            Jacobian = 1.0 
        """
        zeta_Ni = jl_dE / epsilon  # jacobian * jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon

        log_forward_ion[i,j] = np.log10(zeta_Ni)

print("Extremes of zeta(N, \mu=1) forward: ", log_forward_ion[0,0], log_forward_ion[0,-1])

forward_ion_differential = np.transpose(10**log_forward_ion)*diff_mu_ism
f_overall_ionization = np.log10(np.sum(forward_ion_differential, axis=1))

print("log10(zeta_-(s,mu))\n",log_forward_ion)
print("Delta mu\n",diff_mu_ism)
print("zeta(s,mu)*Delta mu\n",forward_ion_differential)
print("log10(zeta(s))\n",f_overall_ionization)

""" 
# backward moving Cosmic Rays 

"""
c = (len(mu_ism), len(BackwardColumn[0,:]))
log_backward_ion  = np.zeros(c)

for i, mui in enumerate(-1*mu_ism):

    for j, Nj in enumerate(BackwardColumn[i,:]): # will go from 3,4,5,6--- almost in integers#

        jl_dE = 0.0

        for k, E in enumerate(energy): 

            Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) # E_i(E, N)
    
            isms  = ism_spectrum(Ei)            # log_10(j_i(E_i))
            llei  = loss_function(Ei)           # log_10(L(E_i))
            jl_dE += isms*llei*diff_energy[k]   # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        """             
            if 1 - (bfield[j]/bfield[0])*(1 - mui**2) > 0:
                mu_local = np.sqrt(1 - (bfield[j]/bfield[-1])*(1 - mui**2))
                Jacobian = (mui/mu_local)*(bfield[j]/bfield[-1])
            else:
                Jacobian = 0.0
                break
            Jacobian = 1.0
         """
        zeta_Ni = jl_dE / epsilon  # jacobian * jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon
        log_backward_ion[i,j] = np.log10(zeta_Ni)

print("Extremes of zeta(N, \mu=1) backward", log_backward_ion[0,0], log_backward_ion[0,-1])

backward_ion_differential = np.transpose(10**log_backward_ion)*diff_mu_ism
b_overall_ionization = np.log10(np.sum(backward_ion_differential, axis=1))

print("log10(zeta_-(s,mu))\n",log_backward_ion)
print("Delta mu\n",diff_mu_ism)
print("zeta(s,mu)*Delta mu\n",backward_ion_differential)
print("log10(zeta_-(s))\n",b_overall_ionization)

"""  
Mirrored Particles

"""
pairs, (index_global_max, upline) = pocket_finder(bfield, 0, plot=False)

# lets make a list of pairs corresponding to regions considered pockets






total_ionization = np.log10(10**f_overall_ionization + 10**b_overall_ionization[::-1])

"""  
\mathcal{L} & \mathcal{H} Model: Protons

"""
model_H = [1.001098610761e7, -4.231294690194e6,  7.921914432011e5,
          -8.623677095423e4,  6.015889127529e3, -2.789238383353e2,
           8.595814402406e0, -1.698029737474e-1, 1.951179287567e-3,
          -9.937499546711e-6
]


model_L = [-3.331056497233e+6,  1.207744586503e+6,-1.913914106234e+5,
            1.731822350618e+4, -9.790557206178e+2, 3.543830893824e+1, 
           -8.034869454520e-1,  1.048808593086e-2,-6.188760100997e-5, 
            3.122820990797e-8]

logzl = []

for i,Ni in enumerate(column_density):
    lzl = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
    logzl.append(lzl)

print("Extremes of fitting zeta(N) ", logzl[0], logzl[-1])

logzetalfit = np.array(logzl)

logzh = []

for i,Ni in enumerate(column_density):
    lzh = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_H)] )
    logzh.append(lzh)

print("Extremes of fitting zeta(N) ", logzh[0], logzh[-1])

logzetahfit = np.array(logzh)

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
        fs_ionization = 4*np.pi*(1 / epsilon) * (1 + d) / (a + 2 * d) * Jstar * Lstar * Estar * I * (Ni / Nof) ** (-gammaf)
        free_streaming_ion.append(np.log10(fs_ionization))
    
    axs[0, 0].plot(column_density, free_streaming_ion, label=f'$log(\zeta_f(N, a={a}))$', linestyle="--", color=c)
    axs[0, 0].text(column_density[-1], free_streaming_ion[-1], f'a={a}, $\gamma_f$={gammaf:.2f}', fontsize=10, color=c)

# Plot Ionization vs Column Density (First plot)
axs[0,0].plot(column_density, logzetalfit, label='$\mathcal{L}$', linestyle="--", color="grey")
axs[0,0].plot(column_density, logzetahfit, label='$\mathcal{H}$', linestyle="--", color="dimgrey")
axs[0,0].plot(column_density, log_ion, label='$log_{10}(\zeta_{\int dE})$', linestyle="-", color="black")
axs[0,0].plot(column_density, svnteen, label='$\zeta = 10^{-17}$', linestyle=":", color="dimgrey")
axs[0,0].set_xscale('log')
axs[0,0].set_title('Ionization vs Column Density')
axs[0,0].set_xlabel('$N cm^{-2}$ (log scale)')
axs[0,0].set_ylabel('$\zeta(N)$ (log scale)')
axs[0,0].legend()
axs[0,0].grid(True)

colors = cm.rainbow(np.linspace(0, 1, len(distance)))

# Plot s vs ionization by both forward and backward moving particles (Second plot) 
#axs[0,1].scatter(distance, total_ionization,label='$\zeta(s)$', linestyle="--", color="skyblue", s=5)
#axs[0,1].scatter(BackwardColumn[0,:], b_overall_ionization,label='$\zeta_-(s)$', marker="v", color="c", s=5)
axs[0,1].scatter(ForwardColumn[0,:], f_overall_ionization,label='$\zeta_+(s)$', marker="^", color="m", s=7)
#axs[0,1].plot(column_density, log_ion, label='$log_{10}(\zeta_{\int dE})$', linestyle="--", color="dimgrey")
axs[0,1].set_xscale('log')
axs[0,1].set_title('Ionizations at the trajectory')
axs[0,1].set_xlabel('s')
axs[0,1].set_ylabel('$\zeta(N)$ (log scale)')
axs[0,1].grid(True)

svnteen = np.ones_like(ForwardColumn[-1,:])*(-15)

colors = cm.rainbow(np.linspace(0, 1, len(mu_ism)))

# Plot Ionization vs Column Density (Third plot)
for i, c in enumerate(colors):
    axs[1,0].scatter(ForwardColumn[i,:], log_forward_ion[i,:], label=f'$\zeta(mu_i={mu_ism[i]})$', marker="|", color=c, s=5)

axs[1,0].plot(column_density, log_ion, label='$log_{10}(\zeta_{\int dE})$', linestyle="--", color="black")
axs[1,0].set_xscale('log')
axs[1,0].set_xlabel('$N(s)$ (log scale)')
axs[1,0].set_ylabel('$\zeta_+(N(s))$ (log scale)')
axs[1,0].grid(True)
#axs[1,0].legend()
#axs[1,0].set_ylim([-16, -15])


svnteen = np.ones_like(BackwardColumn[-1,:])*(-15)
colors = cm.rainbow(np.linspace(0, 1, len(mu_ism)))

# Plot Spectrum L(E) vs Energy (Fourth plot)
for i, c in enumerate(colors):
    axs[1,1].scatter(BackwardColumn[i,:], log_backward_ion[i,:], label=f'$\zeta(mu_i={-mu_ism[i]})$', marker="|", color=c, s=5)

axs[1,1].plot(column_density, log_ion, label='$log_{10}(\zeta_{\int dE})$', linestyle="--", color="black")
axs[1,1].set_xscale('log')
axs[1,1].set_xlabel('$N(s)$ (log scale)')
axs[1,1].set_ylabel('$\zeta_-(N(s))$ (log scale)')
axs[1,1].grid(True)
#axs[1,1].legend()
#axs[1,1].set_ylim([-16, -15])

#fig.tight_layout(pad=2.0)
plt.savefig("b_output_data/b_ionizations.png")
#plt.show()
