import numpy as np
from scipy.interpolate import interp1d

def xsec_simple(Enu):
    return 4e-13*Enu # m2/t, Enu in GeV

def xsecbar_simple(Enu):
    return 2e-13*Enu # m2/t, Enu in GeV

# def xsec(Enu):
#     dat = np.loadtxt("nuOxsec_pernucleon_1e-38.txt", delimiter=",")
#     # nucleons per tonne
#     n_nucleon_per_tonne = 6.022*1e23*1e6
#     scale = 1e-38*1e-4*n_nucleon_per_tonne
#     return np.interp(Enu, dat[0], Enu*dat[1]*scale)

# def xsecbar(Enu):
#     dat = np.loadtxt("nubarOxsec_pernucleon_1e-38.txt", delimiter=",")
#     # nucleons per tonne
#     n_nucleon_per_tonne = 6.022*1e23*1e6
#     scale = 1e-38*1e-4*n_nucleon_per_tonne
#     return np.interp(Enu, dat[0], Enu*dat[1]*scale)



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

if __name__ == "__main__":
    print("Xsec check")

    print("Compare nu at 1 TeV", xsec(1e3), xsec_simple(1e3))
    print("Compare nubar at 1 TeV", xsecbar(1e3), xsecbar_simple(1e3))

    print("Compare nu at 5 TeV", xsec(5e3), xsec_simple(5e3))
    print("Compare nubar at 5 TeV", xsecbar(5e3), xsecbar_simple(5e3))

