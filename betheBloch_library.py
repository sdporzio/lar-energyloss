### ENERGY LOSS DOCUMENTATION
# https://indico.fnal.gov/event/14933/contributions/28526/attachments/17961/22583/Final_SIST_Paper.pdf
# https://pdg.lbl.gov/2009/reviews/rpp2009-rev-passage-particles-matter.pdf
# https://www.physics.princeton.edu/~phy209/week2/bethe_formula.pdf
### CONSTANTS DOCUMENTATION
# https://lar.bnl.gov/properties/

### IMPORTS
import numpy as np
from scipy.interpolate import interp1d

### GENERIC CONSTANTS
c = 299792458 # m/s (speed of light)
c2 = np.power(c,2.) # Squared
z2 = 1. #  Multiples of electron charge (electric charge of moving particle)
ZA = 0.4509 # Argon Z/A (18p / 39.79pn)
rho = 1.396 # g/cm3 (liquid argon density)
m_e = 0.510/c2 # MeV/c2 (electron mass)
### BETHE-BLOCH CONSTANTS
K = 0.307075 # MeV g-1 cm2 (constant)
I = 188*1e-6 # MeV (mean excitation energy of material, usually Z*10 eV)
I2 = np.power(I,2.) # Squared
j = 0.200 # Bichsel j constant, from H. Bichsel, Rev. Mod. Phys.60, 663 (1988)
# Sternheimer's parameters (only for density effect), taken from detectorproperties.fcl (LArSoft)
stern_a = 0.1956
stern_k = 3
stern_x0 = 0.2
stern_x1 = 3
stern_C = 5.2146
### RECOMBINATION CONSTANTS
W = 23.6 # eV needed to produce e-
elPerMeV = 1./W # Number of electrons produced per MeV (42.37 e- per keV, 42,370 per MeV)
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
    dedr = dedx * rho # Bethe-Block is in units of "density-scaled distance", we need actually distance traversed in liquid argon
    return kin,dedr

### MOST PROBABLE VALUE (MPV) FROM LANDAU-VAVILOV-BICHSEL (LVB)
def MPV(mass,energy):
    '''
    Mass and energy in MeV.
    Return MeV/cm. Takes into account lar density already.
    '''

    # Return nothing for nonsensical requests
    if energy<mass:
        return 0,0
    # Define variables
    momentum,gamma,gamma2,beta,beta2,kin,Mass = Relat(mass,energy)
    # Calculate xi
    xi = 1/2.*K*z2*ZA/beta2
    # Define density effect corrections
    stern_x = np.log10(gamma*beta)
    delta = 0
    if stern_x < stern_x0: delta = 0
    if stern_x >= stern_x0 and stern_x < stern_x1: delta = 2.*np.log(10)*stern_x - stern_C + stern_a*np.power(stern_x1-stern_x,stern_k)
    if stern_x >= stern_x1: delta = 2.*np.log(10)*stern_x - stern_C
    # Calculate the different parts
    p1 = np.log(2*m_e*c2*beta2*gamma2/I)
    p2 = np.log(xi/I) + j - beta2 - delta
    dedx = xi * (p1 + p2)
    dedr = dedx * rho # Bethe-Block is in units of "density-scaled distance", we need actually distance traversed in liquid argon
    return kin, dedr


### RESIDUAL RANGES IN CSDA APPROXIMATION
def ResRange(mass,e_init=1000.,step=0.1):
    '''
    Mass and energy in MeV.
    Initial energy is initial kinetic energy
    Step in cm.
    Return MeV/cm. Takes into account lar density already.
    '''

    # Get curves as function of kinetic energy (inefficient, but I am lazy and recycling code)
    en = np.logspace(2,5,10000)
    tab_mean_k = [BetheBloch(mass,e)[0] for e in en]
    tab_mean_dedx = [BetheBloch(mass,e)[1] for e in en]
    tab_mpv_k = [MPV(mass,e)[0] for e in en]
    tab_mpv_dedx = [MPV(mass,e)[1] for e in en]
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

    return mean_dist, mean_dedx, mpv_dist, mpv_dedx


def Recombination(dedx,model='box',efield=0.5):
    '''
    Takes as input dedx as MeV/cm
    Return #e-/cm produced, not collected (you need to take into account lifetime and diff for that)
    Model can be 'box' or 'birks'
    Efield in kV/cm
    '''
    if model=='box':
        xi = box_B/(rho*efield) * dedx
        recomb = np.log(box_A + xi)/xi
        return recomb
    else if model=='birks':
        return birks_A / ( 1. + birks_k/(rho*efield * dedx) )
    else:
        raise Exception("Wrong model. Choose 'box' or 'birks'")
