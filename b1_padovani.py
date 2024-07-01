from z_library import *
import matplotlib.pyplot as plt
import numpy as np

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
Epion = 280e+6


""" Ionzation: Measure of energy/charge deposited from a charged particle in a medium per unit time """

energy = np.array([1.0e+2*(10)**(14*k/size) for k in range(size)])  # Energy values from 1 to 10^15
diff_energy = np.array([(10)**(14*k/size) for k in range(size)])
logenergy = np.log10(energy)                                        # log10(Energy) values from 0 to 15
column_density = np.array([1.0e+19*(10)**(8*k/size) for k in range(size)])

# for one value of \mu only
log_forward_ion = []
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

for i, Ni in enumerate(column_density): # will go from 3,4,5,6--- almost in integers#

    jl_dE = 0.0
    for k, E in enumerate(energy): # will go from 3,4,5,6--- almost in integers#

        if Ni == column_density[-1]:
            proton_local_spectrum.append(log_proton_spectrum(E))         # J_p as in Padovani et al equation (1)
            low_energy_proton_ism_spectrum.append(log_ism_spectrum(E)) # J_i(E) as in Silsbee, 2018 equation  (17)
            log_proton_ism_spectrum.append(log_ism_spectrum(Ei))                    # J_i(E_i) as in Silsbee, 2018 equation  (17)
            log_lossE.append(log_loss_function(E))                       # ==> L(E)
            log_lossEi.append(log_loss_function(Ei))                            # ==> L(Ei) Loss function inside cloud for Entrance energy (Ei)
            log_jp = log_proton_ism_spectrum[-1] + log_lossEi[-1] - log_lossE[-1]   # J_+ as in Silsbee, 2018 equation  (11)
            forward_spectrum.append(log_jp)

        Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Ni)**(1 / (1 + d)) # E_i(E, N)

        isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
        llei = loss_function(Ei)           # log_10(L(E_i))
        jl_dE += isms*llei*diff_energy[k] # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
    zeta_Ni = jl_dE/epsilon/epsilon #jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon
    
    log_forward_ion.append(np.log10(zeta_Ni))


print("Extremes of zeta(N) Padovani et al: ", log_forward_ion[0], log_forward_ion[-1])

"""  
Mosaic With all the plots
"""

model_L = [-3.331056497233e+6, 1.207744586503e+6,-1.913914106234e+5,
            1.731822350618e+4,-9.790557206178e+2, 3.543830893824e+1, 
           -8.034869454520e-1, 1.048808593086e-2,-6.188760100997e-5, 
            3.122820990797e-8]

logz = []

print("Extremes of N ", np.log10(column_density[0]), np.log10(column_density[-1]))

differential_fit = lambda z2,z1,e2,e1: (z2-z1)/(e2-e1) # slope of a log function log(F_2) = m log(e2) + const.
derivative_fit = []

for i,Ni in enumerate(column_density):
    lz = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
    logz.append(lz)


print("Extremes of zeta(N) ",logz[0], logz[-1])

logzetafit = np.array(logz)
svnteen = np.ones_like(column_density)*(-17)



"""  
PLotting 

"""
del a

s_of_m = ["#F4A6D7", "#D10056", "#7D2248"]

fig, axs = plt.subplots(2, 2, figsize=(10,10))  # Create a 2x2 grid of subplots

""" 
Free Streaming Cosmic Rays, Analytical Expression (Silsbee & Ivlev, 2019)
"""

from scipy import integrate

If = {
    "a=0.4": 5.14672,
    "a=0.5": 3.71169,
    "a=0.7": 2.48371
    }

Nof = Estar/((1+d)*Lstar)
epsilon = 1.0e-9
index = 0
for b, I in If.items():
    
    free_streaming_ion = []
    fs_ionization = 0.0
    

    for Ni in column_density:
        
        a = float(b.split("=")[1])
        gammaf = (a+d-1)/(1+d)
        fs_ionization = (1/epsilon)* (1+d) / (a+2*d) * Jstar * Lstar * Estar * (Ni/Nof) ** (-gammaf) * If[b]
        free_streaming_ion.append(np.log(fs_ionization))
        
    axs[0,0].plot(column_density, free_streaming_ion, label=f'$log_{10}(\zeta_f(N, a={a}))$', linestyle="--", color=s_of_m[index])
    index += 1

# Plot Ionization vs Column Density (Second plot)
axs[0,0].plot(column_density, logzetafit, label='$log_{10}(\zeta) \, (\mathrm{Padovani \, et \, al})$', linestyle="--", color="grey")
axs[0,0].plot(column_density, log_forward_ion, label='$log_{10}(\zeta_{\int dE})$', linestyle="-", color="black")
axs[0,0].plot(column_density, svnteen, label='$\zeta = 10^{-17}$', linestyle="--", color="skyblue")
axs[0,0].set_xscale('log')
axs[0,0].set_title('Ionization vs Column Density')
axs[0,0].set_xlabel('Column Density (log scale)')
axs[0,0].set_ylabel('Ionization (log scale)')
axs[0,0].legend()
axs[0,0].grid(True)

# Plot Spectrum j_p vs Energy (Third plot) 
#axs[0,1].plot(energy, proton_local_spectrum, label='$log_{10}(j_p(E)), (Ivlev, 2015)$',linestyle="--", color="dimgrey")
#axs[0,1].plot(energy, low_energy_proton_ism_spectrum, label='$log_{10}(j_i(E)), (Silsbee, 2018)$',linestyle=":", color="red")
axs[0,1].scatter(energy, proton_local_spectrum, marker="|",s=5, color="dimgrey")
axs[0,1].scatter(energy, low_energy_proton_ism_spectrum, marker="|",s=5, color="red")
axs[0,1].set_xscale('log')

axs[0,1].set_title('Spectrum $j_p$ vs Energy')
axs[0,1].set_xlabel('Energy ($log_{10}(E / eV)$)')
axs[0,1].set_ylabel('Spectrum $j_p$')
#axs[0,1].set_ylim([-180, 0])
axs[0,1].legend()
axs[0,1].grid(True)


# Plot Spectrum L(E) vs Energy 
axs[1,0].plot(energy, log_lossEi,label='$log_{10}(L(E_i))$', linestyle="--", color="orange")
axs[1,0].plot(energy, log_lossE,label='$log_{10}(L(E))$', linestyle="--", color="pink")
axs[1,0].scatter(energy, log_lossEi, marker="^", color="orange",s=5)
axs[1,0].scatter(energy, log_lossE, marker="v", color="pink", s=5)
axs[1,0].set_xscale('log')
axs[1,0].set_title('Loss Function vs Energy')
axs[1,0].set_xlabel('Energy ($log_{10}(E / eV)$)')
axs[1,0].set_ylabel('Spectrum $j_p(E)$')
axs[1,0].legend()
axs[1,0].grid(True)

# Plot Ei(E) vs Energy 
axs[1, 1].plot(energy, (energy**(1 + d) + (1 + d) * Lstar* Estar**(d)* column_density[-1])**(1 / (1 + d)), label='$log_{10}(E_i(E, N=10^{27}))$', linestyle="-", color="red")
axs[1, 1].plot(energy, (energy**(1 + d) + (1 + d) * Lstar* Estar**(d)* column_density[0])**(1 / (1 + d)), label='$log_{10}(E_i(E, N==10^{19}))$', linestyle="-", color="red")
axs[1, 1].plot(energy, diff_energy, label='$log_{10}(\Delta(E))$', linestyle="--", color="dimgrey")
axs[1, 1].scatter(energy, diff_energy, marker="x", color="black", s=5)
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')
axs[1, 1].set_title('$log_{10}(f_j(E))$ vs Energy')
axs[1, 1].set_xlabel('Energy ($log_{10}(E / eV)$)')
axs[1, 1].legend()
axs[1, 1].grid(True)

fig.tight_layout(pad=2.0)
plt.savefig("b_output_data/Mosaic.png")

#plt.show()
