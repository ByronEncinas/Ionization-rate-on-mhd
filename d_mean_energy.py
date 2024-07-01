import numpy as np

cross_sections = np.array(np.load("d_input_data/cross_pH2_rel_1e18.npz", mmap_mode='r'))
pLoss = np.array(np.load("d_input_data/Kedron_pLoss.npz", mmap_mode='r'))

print(cross_sections.shape)
print(cross_sections)
print(pLoss.shape)
print(pLoss)

global energy_range

energy_range = (1.0e+6, 1.0e+9)

"""  
Let us remember that the relationship between 
the pLoss and the Cross Section is CS ~ constant* pLoss

\epsilon * \sigma = L(E)

So 

\epsilon = L(E)/\sigma
"""

# extract values of energy, pLossH & pLossfull
with np.load("d_input_data/Kedron_pLoss.npz") as dp:

    pEnergies    = []
    pLossH       = []
    pLossfull    = []
    
    print(dp["E"].shape)
    mid = len(dp["E"])//2
    print(dp["E"][1],dp["E"][mid] , dp["E"][-1])

    """  
    dp data goes from 0.00010008062376766876 eV => 31635521770261.6e+31 eV
    """
    first_third = len(dp["E"])//3
    secon_third = 2*first_third
    
    dE = dp["E"][first_third//2:secon_third]
    dlossH = dp["L_H"][first_third//2:secon_third]
    dlossFull = dp["L_full"][first_third//2:secon_third]

    for i in range(len(dE)):
            
        if (dE[i] > energy_range[0]) and (dE[i] < energy_range[1]):
            pEnergies.append(dE[i]) # guardo la energia y el indice en el que está
            pLossH.append(dlossH[i])
            pLossfull.append(dlossFull[i])
        else:
            pass

# extract values of energy & cross sections
with np.load("d_input_data/cross_pH2_rel_1e18.npz") as dc:

    cEnergies    = []
    CrossSection = []
    
    for i in range(len(dc["E"])):
    
        if (dc["E"][i] > energy_range[0]) and (dc["E"][i] < energy_range[1]):

            cEnergies.append(dc["E"][i]) # guardo la energia y el indice en el que está
            CrossSection.append(dc["sigmap"][i])

# scales and differences between data gathered
print("cE: ",len(cEnergies), cEnergies[0], cEnergies[-1])
print("C-Section: ",len(CrossSection), CrossSection[0], CrossSection[-1])
print("pE: ",len(pEnergies), pEnergies[0], pEnergies[-1])
print("pLoss: ",len(pLossH), pLossH[0], pLossH[-1])

delta = 1/10_000

# Match energies up to 1 thousand percent tolerance
def Match_and_find(Ecross, Eploss, tolerance):
    """  
    find matches within a tolerance in the following form
    Ep - \delta < Ec < Ep + \delta
    Ecross is tuple with values (Ec, c_index)
    Eploss is tuple with values (Ep, p_index)
    """
    # save energies and indices(for later use in)
    energy_et_index = [] # tuples inside
    cross = 0.0
    ploss = 0.0
    epsilon = []

    for i, Ec in enumerate(Ecross): # c_tup = (Ec, c_index)
        
        for j, Ep in enumerate(Eploss): # p_tup = (Ep, p_index)
        
            if Ec > Ep*(1 - tolerance)  and Ec < Ep*(1 + tolerance):
                # (i,j) is the position of (Ec,Ep) at his corresponding database
                energy_et_index.append([(Ec, Ep), (i,j)])

                cross = CrossSection[i]
                ploss = pLossH[j]
                ei = ploss/cross

                epsilon.append(ei) # gathered all values of epsilon calculated
                #print([(Ec + Ep)/2.,  (ploss, cross, ei)]) 

    return epsilon

calculation = Match_and_find(cEnergies, pEnergies, delta) # contains (Ep: tuple, Ec:tuple)

mean_energy = sum(calculation)/len(calculation)
print(mean_energy, 1/mean_energy)
