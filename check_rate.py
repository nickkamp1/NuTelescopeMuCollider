from flux import *
from xsec import *

Etest = 1e3 # GeV
Gtest = Etest / MU_MASS

# Dump

# Making this number up
Lmuontest = 10 # m

Nmuon_dump = MU_PER_BUNCH*REPRATE*S2YR*(Lmuontest/C_M_PER_US)/(MU_TAU*Gtest)

# Accelerator

# 30cm straight section from https://arxiv.org/abs/2407.12450, pointed
# at the telescope 1% of the time (making that up)
Lmuontest = 0.3*0.01 # m

Nmuon_accelerator = MU_PER_BUNCH*(1e6*C_M_PER_US/CIRCUMFERENCE)*S2YR*(Lmuontest/C_M_PER_US)/(MU_TAU*Gtest)

print("Number of muons [e12]:", Nmuon_dump/1e12, Nmuon_accelerator/1e12)

# Example volume, from P ONE
# https://pos.sissa.it/444/1175/pdf
VOLUME = 1e3*np.pi*(5e2)**2
DENSITY = 1 # t / m^3

intrate = xsec(Etest)*DENSITY*np.power(VOLUME, 1./3.)

print("Number of interactions [1e6] / yr:", Nmuon_dump*intrate/1e6, Nmuon_accelerator*intrate/1e6)
print("Number of interactions / s:", Nmuon_dump*intrate/S2YR, Nmuon_accelerator*intrate/S2YR)

print("Number of interactions / spill:", Nmuon_dump*intrate/(REPRATE*S2YR), Nmuon_accelerator*intrate/((1e6*C_M_PER_US/CIRCUMFERENCE)*S2YR))
