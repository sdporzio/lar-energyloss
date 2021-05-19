### ENERGY LOSS DOCUMENTATION
# https://indico.fnal.gov/event/14933/contributions/28526/attachments/17961/22583/Final_SIST_Paper.pdf
# https://pdg.lbl.gov/2009/reviews/rpp2009-rev-passage-particles-matter.pdf
# https://www.physics.princeton.edu/~phy209/week2/bethe_formula.pdf
### CONSTANTS DOCUMENTATION
# https://lar.bnl.gov/properties/
# https://pdg.lbl.gov/2017/AtomicNuclearProperties/HTML/liquid_argon.html


### IMPORTS
import numpy as np
from scipy.interpolate import interp1d

### GENERIC CONSTANTS
c = 299792458 # m/s (speed of light)
c2 = np.power(c,2.) # Squared
z2 = 1. #  Multiples of electron charge (electric charge of moving particle)
m_e = 0.510/c2 # MeV/c2 (electron mass)
### BETHE-BLOCH CONSTANTS
K = 0.307075 # MeV g-1 cm2 (constant)
j = 0.200 # Bichsel j constant, from H. Bichsel, Rev. Mod. Phys.60, 663 (1988)

### LIQUID ARGON ENERGY-LOSS CONSTANTS
### https://pdg.lbl.gov/2017/AtomicNuclearProperties/HTML/liquid_argon.html
### https://pdg.lbl.gov/2017/AtomicNuclearProperties/MUE/muE_liquid_argon.pdf
ZA = 0.4509 # Argon Z/A (18p / 39.79pn)
rho = 1.396 # g/cm3 (liquid argon density)
I = 188*1e-6 # MeV (mean excitation energy of LAr material, usually Z*10 eV)
I2 = np.power(I,2.) # Squared
# Sternheimer's parameters (only for density effect)
stern_a = 0.1956
stern_k = 3
stern_x0 = 0.2
stern_x1 = 3
stern_C = 5.2146

# ### SILICON ENERGY-LOSS CONSTANTS
# ### https://pdg.lbl.gov/2020/AtomicNuclearProperties/HTML/silicon_Si.html
# ### https://pdg.lbl.gov/2020/AtomicNuclearProperties/MUE/muE_silicon_Si.pdf
# ZA = 0.5 # Silicon Z/A (14p / 28.08pn)
# rho = 2.329  # g/cm3 (silicon density)
# I = 173.0*1e-6  # MeV (mean excitation energy of silicon material)
# I2 = np.power(I,2.) # Squared
# # Sternheimer's parameters (only for density effect)
# stern_a = 0.14921
# stern_k = 3.2546
# stern_x0 = 0.2015
# stern_x1 = 2.8716
# stern_C = 4.4355

### LIQUID ARGON RECOMBINATION CONSTANTS
W = 23.6 # eV needed to produce e-
elPerMeV = 1./W * 1e6 # Number of electrons produced per MeV (42.37 e- per keV, 42,370 per MeV)
birks_A = 0.800 # Birks recombination model, parameter A (dim.less)
birks_k = 0.0486 # Birks recombination model, parameter k (kV/cm)*(g/cm2)/MeV
box_A = 0.930 # Modified Box recombination model, parameter A (dim.less)
box_B = 0.212 # Modified Box recombination model, parameter B (kV/cm)*(g/cm2)/MeV


### RELATIVISTIC QUANTITIES
def Relat(mass,energy):
    '''
    Mass and energy in MeV.
    '''

    momentum = np.sqrt(np.power(energy,2.) - np.power(mass,2.))
    gamma = energy/mass
    beta = momentum/energy
    gamma_alt = energy/mass
    beta_alt = momentum/energy
    kin = energy - mass
    beta2 = np.power(beta,2.)
    gamma2 = np.power(gamma,2.)
    Mass = mass/c2
    return momentum,gamma,gamma2,beta,beta2,kin,Mass

### MAX KINETIC ENERGY
def Tmax(mass, energy):
    '''
    Mass and energy in MeV.
    '''

    momentum,gamma,gamma2,beta,beta2,kin,Mass = Relat(mass,energy)
    f1 = 2.*m_e*c2*beta2*gamma2
    f2 = 1. + 2.*gamma*m_e/Mass + np.power(m_e/Mass,2.)
    tmax = f1/f2
    return tmax

### MEAN ENERGY FROM BETHE-BLOCH
def BetheBloch(mass,energy):
    '''
    Mass and energy in MeV.
    Return MeV/cm. Takes into account lar density already.
    '''

    # Return nothing for nonsensical requests
    if energy<mass:
        return 0,0
    # Define variables
    momentum,gamma,gamma2,beta,beta2,kin,Mass = Relat(mass,energy)
    tmax = Tmax(mass,energy)
    # Define density effect corrections
    stern_x = np.log10(gamma*beta)
    delta = 0
    if stern_x < stern_x0: delta = 0
    if stern_x >= stern_x0 and stern_x < stern_x1: delta = 2.*np.log(10)*stern_x - stern_C + stern_a*np.power(stern_x1-stern_x,stern_k)
    if stern_x >= stern_x1: delta = 2.*np.log(10)*stern_x - stern_C
    # Calculate the different parts
    scal = K*z2*ZA/beta2
    p1 = 1/2. * np.log(2.*m_e*c2*beta2*gamma2*tmax/I2)
    p2 = beta2
    p3 = delta/2.
    dedx = scal * (p1 - p2 - p3)
    # Bethe-Block is in units of "density-scaled distance"
    # so dedx is actually (MeV g-1 cm2)
    # we need actually distance traversed in liquid argon
    # Multiplying by density (g cm-3) gives us (Mev cm-1) or MeV/cm
    # which is what we need
    dedr = dedx * rho # Bethe-Bloch is in units of "density-scaled distance", we need actually distance traversed in liquid argon
    return kin, dedr

### MEAN ENERGY FROM BETHE-BLOCH. CALCULATE WHOLE CURVE.
def BetheBlochCurve(mass=105.65):
    '''
    Mass in MeV
    Return MeV/cm. Takes into account lar density already.
    '''
    # Energy range
    erange = np.logspace(2,6,int(1e4))
    kin, dedr = [], []
    for e in erange:
        this_kin, this_dedr = BetheBloch(mass,e)
        kin.append(this_kin)
        dedr.append(this_dedr)
    kin = np.array(kin)
    dedr = np.array(dedr)
    return kin, dedr


### MOST PROBABLE VALUE (MPV) FROM LANDAU-VAVILOV-BICHSEL (LVB)
def Landau(mass,energy,dl):
    '''
    Mass and energy in MeV.
    dl (thickness length) in cm
    Return MeV/cm. Takes into account lar density already.
    '''
    # Thickness is g cm-2, equal to dl multiplied by density
    thickness = dl*rho
    # print(thickness*1000)

    # Return nothing for nonsensical requests
    if energy<mass:
        return 0,0
    # Define variables
    momentum,gamma,gamma2,beta,beta2,kin,Mass = Relat(mass,energy)
    # Calculate xi
    xi = K/2.*ZA*(thickness/beta2)
    # Define density effect corrections
    stern_x = np.log10(gamma*beta)
    delta = 0
    if stern_x < stern_x0: delta = 0
    if stern_x >= stern_x0 and stern_x < stern_x1: delta = 2.*np.log(10)*stern_x - stern_C + stern_a*np.power(stern_x1-stern_x,stern_k)
    if stern_x >= stern_x1: delta = 2.*np.log(10)*stern_x - stern_C
    # Calculate the different parts
    p1 = np.log(2.*m_e*c2*beta2*gamma2/I)
    p2 = np.log(xi/I) + j - beta2 - delta
    deltap_x = xi * (p1 + p2)/thickness
    # Again, deltap over x is is in units of "density-scaled distance"
    # so deltap_x is actually (MeV g-1 cm2)
    # we need actually distance traversed in liquid argon
    # Multiplying by density (g cm-3) gives us (Mev cm-1) or MeV/cm
    # which is what we need
    deltap_r = deltap_x*rho
    return kin, deltap_r

### MPV ENERGY FROM LANDAU-VASILOV CALCULATE WHOLE CURVE.
def LandauCurve(mass=105.65,dl=1):
    '''
    Mass in MeV
    dl (thickness length) in cm
    Return MeV/cm. Takes into account lar density already.
    '''
    # Energy range
    erange = np.logspace(2,6,int(1e4))
    kin, dpdr = [], []
    for e in erange:
        this_kin, this_dpdr = Landau(mass,e,dl)
        kin.append(this_kin)
        dpdr.append(this_dpdr)
    kin = np.array(kin)
    dpdr = np.array(dpdr)
    return kin, dpdr

### RESIDUAL RANGES IN CSDA APPROXIMATION
def ResRange(mass,dl=1,e_init=1000.,step=0.1,outType='dedx',recModel='box'):
    '''
    Mass and energy in MeV.
    Initial energy is initial kinetic energy
    Step in cm.
    dl (thickness length of Landau) in cm
    Return MeV/cm by default (dedx). Takes into account lar density already.
    If 'dqdx' is selected, return number of electrons per cm. You also need to select model ('box' or 'birks')
    '''

    # Get curves as function of kinetic energy (inefficient, but I am lazy and recycling code)
    en = np.logspace(2,5,10000)
    tab_mean_k, tab_mean_dedx = BetheBlochCurve(mass)
    tab_mpv_k, tab_mpv_dedx = LandauCurve(mass,dl)
    f_mean = interp1d(tab_mean_k,tab_mean_dedx)
    f_mpv = interp1d(tab_mpv_k,tab_mpv_dedx)
    
    # Start with initial energy and keep track of distance travelled
    mean_dist, mean_en, mean_dedx = [], [], []    
    curr_en = e_init
    curr_dist = 0.
    # Keep subtracting energy until particle stops
    while curr_en>0:
        e_lost = f_mean(curr_en)
        curr_dist = curr_dist + step
        curr_en = curr_en - e_lost*step
        mean_dist.append(curr_dist)
        mean_en.append(curr_en)
        mean_dedx.append(e_lost)
    # Numpify
    mean_dist = np.array(mean_dist)
    mean_en = np.array(mean_en)
    mean_dedx = np.array(mean_dedx)
    # Distance as distance from end point
    mean_dist = -1*(mean_dist - mean_dist.max())

    # Start with initial energy and keep track of distance travelled
    mpv_dist, mpv_en, mpv_dedx = [], [], []
    curr_en = e_init
    curr_dist = 0.
    # Keep subtracting energy until particle stops
    while curr_en>0:
        e_lost = f_mpv(curr_en)
        curr_dist = curr_dist + step
        curr_en = curr_en - e_lost*step
        mpv_dist.append(curr_dist)
        mpv_en.append(curr_en)
        mpv_dedx.append(e_lost)
    # Numpify
    mpv_dist = np.array(mpv_dist)
    mpv_en = np.array(mpv_en)
    mpv_dedx = np.array(mpv_dedx)
    # Distance as distance from end point
    mpv_dist = -1*(mpv_dist - mpv_dist.max())

    if outType=='dqdx':
        mean_dqdx = np.array([dedx*elPerMeV*Recombination(dedx,model=recModel) for dedx in mean_dedx])
        mpv_dqdx = np.array([dedx*elPerMeV*Recombination(dedx,model=recModel) for dedx in mpv_dedx])
        return mean_dist, mean_dqdx, mpv_dist, mpv_dqdx
    else:
        return mean_dist, mean_dedx, mpv_dist, mpv_dedx


### CALCULATE RECOMBINATION FACTOR BOTH IN BOX AND BIRKS MODEL
### https://arxiv.org/pdf/1306.1712.pdf
def Recombination(dedx,model='box',efield=0.5):
    '''
    Takes as input dedx as MeV/cm
    Return #e-/cm produced, not collected (you need to take into account lifetime and diff for that)
    Model can be 'box' or 'birks'
    Efield in kV/cm
    '''
    # Protect against negative values
    if dedx<=0:
        return 0

    # Return modified box model
    # dedx must be in MeV cm2 g-1, so we need to redivide by density
    if model=='box':
        xi = box_B/efield * dedx/rho
        recomb = np.log(box_A + xi)/xi
        # Make sure value is positive
        if recomb>0: return recomb
        else: return 0

    # Return Birks model
    # dedx must be in MeV cm2 g-1, so we need to redivide by density
    elif model=='birks':
        recomb = birks_A / ( 1. + birks_k/efield * dedx/rho )
        # Make sure value is positive
        if recomb>0: return recomb
        else: return 0

    # Make sure model input is correct
    else:
        raise Exception("Wrong model. Choose 'box' or 'birks'")


dedx = np.linspace(0,10,500)
dqdx_box = [de*elPerMeV*Recombination(de,model='box') for de in dedx]
dqdx_birks = [de*elPerMeV*Recombination(de,model='birks') for de in dedx]
box_inv = interp1d(dqdx_box,dedx,fill_value='extrapolate')
birks_inv = interp1d(dqdx_birks,dedx,fill_value='extrapolate')

def ConvertQtoE(dqdx,model='box'):
    '''
    Takes as input dqdx as e-/cm and return MeV/cm
    '''
    # Return modified box model
    if model=='box':
        return box_inv(dqdx)
    # Return Birks model
    elif model=='birks':
        return birks_inv(dqdx)
    # Make sure model input is correct
    else:
        raise Exception("Wrong model. Choose 'box' or 'birks'")
