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


DIS_nu_CC = interp1d(DIS_nu_CC_data[:,0],DIS_nu_CC_data[:,1],fill_value="extrapoloate")
DIS_nu_NC = interp1d(DIS_nu_NC_data[:,0],DIS_nu_NC_data[:,1],fill_value="extrapoloate")
DIS_nubar_CC = interp1d(DIS_nubar_CC_data[:,0],DIS_nubar_CC_data[:,1],fill_value="extrapoloate")
DIS_nubar_NC = interp1d(DIS_nubar_NC_data[:,0],DIS_nubar_NC_data[:,1],fill_value="extrapoloate")

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

# Charm dimuons from https://arxiv.org/pdf/2408.05866

WCG_data = np.loadtxt("data/WCG_DIS_isoscalar.txt",delimiter=",")
pb_to_cm2 = 1e-36
charm_fragmentation = {"D0":0.66,
                       "D":0.26,
                       "Lambda":0.08}
muon_BR_per_hadron = {"D0":0.067,
                       "D":0.176,
                       "Lambda":0.035}
charm_muon_BR = sum([charm_fragmentation[key]*muon_BR_per_hadron[key] for key in charm_fragmentation.keys()])


WCG_nu_CC_charm = interp1d(np.log10(WCG_data[:,0]),np.log10(WCG_data[:,1]*WCG_data[:,3]*pb_to_cm2))
WCG_nubar_CC_charm = interp1d(np.log10(WCG_data[:,0]),np.log10(WCG_data[:,6]*WCG_data[:,8]*pb_to_cm2))

def xsec_charm_dimuon(Enu):
    return 10**WCG_nu_CC_charm(np.log10(Enu)) * cm2_per_nucleon__to__m2_per_ton * charm_muon_BR

def xsecbar_charm_dimuon(Enu):
    return 10**WCG_nubar_CC_charm(np.log10(Enu)) * cm2_per_nucleon__to__m2_per_ton * charm_muon_BR

# Tridents from https://github.com/beizhouphys/neutrino-W-boson-and-trident-production/tree/master

trident_data = np.loadtxt("data/vmO16TOvmmMX_tot.dat")
trident_CC = interp1d(np.log10(trident_data[:,0]),np.log10(trident_data[:,1]/16)) # 16 for nucleons/O16

def xsec_trident_dimuon(Enu):
    return 10**trident_CC(np.log10(Enu)) * cm2_per_nucleon__to__m2_per_ton

def xsecbar_trident_dimuon(Enu):
    return 10**trident_CC(np.log10(Enu)) * cm2_per_nucleon__to__m2_per_ton

# W boson production from https://github.com/beizhouphys/neutrino-W-boson-and-trident-production/tree/master

WBP_data = np.loadtxt("data/numu_H2O_TO_mu_W_X_tot.txt")
WBP_CC = interp1d(np.log10(WBP_data[:,0]),np.log10(WBP_data[:,1]/18),fill_value="extrapolate") # 18 for nucleons/H20

def xsec_WBP_dimuon(Enu):
    return 10**WBP_CC(np.log10(Enu)) * cm2_per_nucleon__to__m2_per_ton

def xsecbar_WBP_dimuon(Enu):
    return 10**WBP_CC(np.log10(Enu)) * cm2_per_nucleon__to__m2_per_ton

if __name__ == "__main__":
    print("Xsec check")

    print("Compare nu at 1 TeV", xsec(1e3), xsec_simple(1e3))
    print("Compare nubar at 1 TeV", xsecbar(1e3), xsecbar_simple(1e3))

    print("Compare nu at 5 TeV", xsec(5e3), xsec_simple(5e3))
    print("Compare nubar at 5 TeV", xsecbar(5e3), xsecbar_simple(5e3))

