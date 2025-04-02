import numpy as np
from scipy.interpolate import interp1d

DIS_nu_CC_data = np.loadtxt("data/DIS_cross_section_tables_CC_nu.txt")
DIS_nu_NC_data = np.loadtxt("data/DIS_cross_section_tables_NC_nu.txt")
DIS_nubar_CC_data = np.loadtxt("data/DIS_cross_section_tables_CC_nubar.txt")
DIS_nubar_NC_data = np.loadtxt("data/DIS_cross_section_tables_NC_nubar.txt")


DIS_nu_CC = interp1d(DIS_nu_CC_data[:,0],DIS_nu_CC_data[:,1])
DIS_nu_NC = interp1d(DIS_nu_NC_data[:,0],DIS_nu_NC_data[:,1])
DIS_nubar_CC = interp1d(DIS_nubar_CC_data[:,0],DIS_nubar_CC_data[:,1])
DIS_nubar_NC = interp1d(DIS_nubar_NC_data[:,0],DIS_nubar_NC_data[:,1])

cm2_per_nucleon__to__m2_per_ton = (1e-2)**2 * 1e6 * 6.02e23


def xsec(Enu):
    return DIS_nu_CC(Enu) * cm2_per_nucleon__to__m2_per_ton # m2/t, Enu in GeV
    #return 4e-13*Enu # m2/t, Enu in GeV

def xsecbar(Enu):
    return DIS_nubar_CC(Enu) * cm2_per_nucleon__to__m2_per_ton # m2/t, Enu in GeV
    #return 2e-13*Enu # m2/t, Enu in GeV

def xsec_NC(Enu):
    return DIS_nu_NC(Enu) * cm2_per_nucleon__to__m2_per_ton # m2/t, Enu in GeV

def xsecbar_NC(Enu):
    return DIS_nubar_NC(Enu) * cm2_per_nucleon__to__m2_per_ton # m2/t, Enu in GeV
