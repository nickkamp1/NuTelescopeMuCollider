import numpy as np
from scipy.integrate import quad, nquad
import flux

Emu = 1000 # GeV
Pmu = 1 # Polarization

def integrand(fluxf, Enu, costh):
    return 2*np.pi*fluxf(Emu, 1, Pmu, Enu, costh, 1)

options={'limit':1000}

# low muon energy, integrate over the full range
if Emu < 1:
    ranges = [(0, Emu), (-1, 1)]
# high muon energy, restrict to where the support is
else: 
    ranges = [(0, Emu), (1-1./Emu, 1)]

# print(quad(lambda t: flux.costhcm_dcosthlab(Emu, t), *ranges[1], **options))
print(nquad(lambda e,c: integrand(flux.numu_flux, e, c), ranges, opts=[{}, options]))
print(nquad(lambda e,c: integrand(flux.nue_flux, e, c), ranges, opts=[{}, options]))
