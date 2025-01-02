import numpy as np

# constants
MU_MASS = 0.1057 # GeV
MU_TAU = 2.2 # us
C_M_PER_US = 299.792 # m / us

# Two types of possible muon beam configurations:
#
# 1) A section of the accelerator points at the target. 
#    In this case, the rate is equal to the rotation rate
#    around the accelerator.
#
# 2) The beam dump points at the target. In this case, the
#    rate is equal to the repition rate (assuming the rate
#    of muons in is equal to the rate of muons out).

# Parameters taken from: https://arxiv.org/abs/2407.12450
# Tables 1.1 and 6.4

MU_PER_BUNCH = 1.8e12
REPRATE = 5 # Hz
S2YR = 525600.*60. # moments so dear...
TIME = S2YR*1 # one year
CIRCUMFERENCE = 10e3 # m

def numu_flux_accelerator(Emuon, Lmuon, Pmuon, enu, costhlab, baseline):
    return nu_flux_accelerator(numu_flux, Lmuon, Pmuon, enu, costhlab, baseline)
def nue_flux_accelerator(Emuon, Lmuon, Pmuon, enu, costhlab, baseline):
    return nu_flux_accelerator(nue_flux, Lmuon, Pmuon, enu, costhlab, baseline)

def numu_flux_dump(Emuon, Lmuon, Pmuon, enu, costhlab, baseline):
    return nu_flux_dump(numu_flux, Lmuon, Pmuon, enu, costhlab, baseline)
def nue_flux_dump(Emuon, Lmuon, Pmuon, enu, costhlab, baseline):
    return nu_flux_dump(nue_flux, Lmuon, Pmuon, enu, costhlab, baseline)

def nu_flux_accelerator(fluxf, Emuon, Lmuon, Pmuon, enu, costhlab, baseline):
    G = Emuon / MU_MASS
    Nmuon = MU_PER_BUNCH*(C_M_PER_US/CIRCUMFERENCE)*S2YR*(Lmuon/C_M_PER_US)/(MU_TAU*G) 
    return fluxf(Emuon, Nmuon, Pmuon, enu, costhlab, baseline)

def nu_flux_dump(fluxf, Emuon, Lmuon, Pmuon, enu, costhlab, baseline):
    G = Emuon / MU_MASS
    Nmuon = MU_PER_BUNCH*REPRATE*S2YR*(Lmuon/C_M_PER_US)/(MU_TAU*G)
    return fluxf(Emuon, Nmuon, Pmuon, enu, costhlab, baseline)

def costhcm_v(Emuon, costhlab):
    G = Emuon/MU_MASS
    B = np.sqrt(1 - 1/G**2)
    return (B - costhlab) / (B*costhlab-1)

def numu_flux(Emuon, Nmuon, Pmuon, enu, costhlab, baseline):
    G = Emuon/MU_MASS
    B = np.sqrt(1 - 1/G**2)
    
    costhcm = costhcm_v(Emuon, costhlab)
    Emax = Emuon*(1 + B*costhcm)/2
    x = enu/Emax
    
    M = 1 / (G*(1-B*costhlab))
    
    fluence = M**2*(2*x**2/(4*np.pi))*((3-2*x) + (1-2*x)*Pmuon*costhcm)
    if not isinstance(fluence, float):
        fluence[x > 1] = 0
    elif x > 1:
        fluence = 0
    return Nmuon*fluence/Emax/baseline**2

def nue_flux(Emuon, Nmuon, Pmuon, enu, costhlab, baseline):
    G = Emuon/MU_MASS
    B = np.sqrt(1 - 1/G**2)
    
    costhcm = costhcm_v(Emuon, costhlab)
    Emax = Emuon*(1 + B*costhcm)/2
    x = enu/Emax
    
    M = 1 / (G*(1-B*costhlab))
    
    fluence = M**2*(12*x**2/(4*np.pi))*((1-x) + (1-x)*Pmuon*costhcm)
    if not isinstance(fluence, float):
        fluence[x > 1] = 0
    elif x > 1:
        fluence = 0
    return Nmuon*fluence/Emax/baseline**2
