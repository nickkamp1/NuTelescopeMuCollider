import numpy as np

####################
# Global Variables
####################

# Neutrino oscillation parameters
th12 = 33.44 *np.pi/180
th23 = 49.20 *np.pi/180
th13 =  8.57 *np.pi/180

dm21 = 7.42e-5 # eV^2
dm31 = 2.517e-3 # eV^2

GeV = 1#1e9

# matter effect constants
Ye = 0.5 # e - / nucleon
R_earth = 6371.0 # km
GFermi = 1.166e-23 # eV^-2
NA = 6.02e23
hbarc = 197.3e-7 # eV cm
nucleon_mass_density_to_electron_number_density = Ye * NA * hbarc**3 # nucleons / cm^3 -> e- eV^3

# PREM model
DPdepth = np.genfromtxt("DensityProfileDepthAK135.txt")
DPdensity = np.genfromtxt("DensityProfileDensityAK135.txt")


# Possible Baselines

# ICECUBE

# Radius of earth
R_earth = 6371.0 # km

# Chicago latitude
fermi_lat = 41.88
fermi_long = 88.26
alpha = ((90 - fermi_lat)/2)*np.pi/180

# IceCube Geometry
ICECUBE_lat = -90
ICECUBE_long = 0 # arbitrary
ICECUBE_BASELINE = np.sqrt(2*R_earth**2*(1 - np.cos((90+fermi_lat)*np.pi/180)))*1e3 # meters

# PONE Geometry
PONE_lat = 47.52
PONE_long = -131.61
PONE_BASELINE = 1758*1.60934*1e3

# KM3NeT Geometry
KM3NeT_lat = 36.27
KM3NeT_long = 16.10
KM3NeT_BASELINE = 4427*1.60934*1e3


experiment_list = ["KM3","IC","PONE"]
baseline_list = {"PONE":PONE_BASELINE,
                 "KM3":KM3NeT_BASELINE,
                 "IC":ICECUBE_BASELINE}
latitude_list = {"PONE":PONE_lat,
                 "KM3":KM3NeT_lat,
                 "IC":ICECUBE_lat}
longitude_list = {"PONE":PONE_long,
                 "KM3":KM3NeT_long,
                 "IC":ICECUBE_long}
color_list = {"PONE":"crimson",
              "KM3":"fuchsia",
              "IC":"deepskyblue",}

# 500 mega-tonne mass, baseline
MASS = 500e6

# density of water
DENSITY = 1 # t / m^3

VOLUME = MASS / DENSITY
RADIUS = np.power(VOLUME/((4./3.)*np.pi), 1./3.)
Emuon = 5e3

# Unpolarized
P = 0

deltaCP0 = 0.