import numpy as np

def xsec_simple(Enu):
    return 4e-13*Enu # m2/t, Enu in GeV

def xsecbar_simple(Enu):
    return 2e-13*Enu # m2/t, Enu in GeV

def xsec(Enu):
    dat = np.loadtxt("nuOxsec_pernucleon_1e-38.txt", delimiter=",")
    # nucleons per tonne
    n_nucleon_per_tonne = 6.022*1e23*1e6
    scale = 1e-38*1e-4*n_nucleon_per_tonne
    return np.interp(Enu, dat[0], Enu*dat[1]*scale)

def xsecbar(Enu):
    dat = np.loadtxt("nubarOxsec_pernucleon_1e-38.txt", delimiter=",")
    # nucleons per tonne
    n_nucleon_per_tonne = 6.022*1e23*1e6
    scale = 1e-38*1e-4*n_nucleon_per_tonne
    return np.interp(Enu, dat[0], Enu*dat[1]*scale)

if __name__ == "__main__":
    print("Xsec check")

    print("Compare nu at 1 TeV", xsec(1e3), xsec_simple(1e3))
    print("Compare nubar at 1 TeV", xsecbar(1e3), xsecbar_simple(1e3))

    print("Compare nu at 5 TeV", xsec(5e3), xsec_simple(5e3))
    print("Compare nubar at 5 TeV", xsecbar(5e3), xsecbar_simple(5e3))

