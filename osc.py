try:
    import oschelper
except:
    raise ImportError("You must build the oschelper library first!")

import numpy as np

# Neutrino oscillation parameters
th12 = 33.44 *np.pi/180
th23 = 49.20 *np.pi/180
th13 =  8.57 *np.pi/180

dm21 = 7.42e-5
dm31 = 2.517e-3

# matter effect
Ye = 0.5
R_earth = 6371.0 # km

# PREM model
DPdepth = np.genfromtxt("DensityProfileDepthAK135.txt")
DPdensity = np.genfromtxt("DensityProfileDensityAK135.txt")

deltaCP0 = 0.

def oscillate(E, baseline, nuind, nusign, dL=1e2, # m 
              rhoset=None, rhoshift=0, rhoscale=1,
              th12=th12, th23=th23, th13=th13, dm21=dm21, dm31=dm31, deltaCP=0,
              epsee=0, epsemu=0, epsetau=0, epsmumu=0, epsmutau=0, epstautau=0):

    if isinstance(E, float) or isinstance(E, int):
        E = np.array([E]).astype(float)
    
    c12 = np.cos(th12)
    c23 = np.cos(th23)
    c13 = np.cos(th13)
    s12 = np.sin(th12)
    s23 = np.sin(th23)
    s13 = np.sin(th13)
    
    # Vacuum oscillations
    U = np.array(
    [
        [c13*c12, c13*s12, s13*np.exp(-1j*deltaCP)],
        [-c23*s12-s13*s23*c12*np.exp(1j*deltaCP), c23*c12-s13*s23*s12*np.exp(1j*deltaCP), c13*s23],
        [s23*s12-s13*c23*c12*np.exp(1j*deltaCP), -s23*c12-s13*c23*s12*np.exp(1j*deltaCP), c13*c23]
    ])
    # Matter Potential
    # TODO: What is going on with this factor if 2?
    MP = 2*nusign*np.array([
        [1+epsee, epsemu, epsetau],
        [np.conj(epsemu), epsmumu, epsmutau],
        [np.conj(epsetau), np.conj(epsmutau), epstautau]
    ])

    if nusign < 0:
        U = np.conj(U)
    
    steps = np.linspace(0, baseline, int(baseline/dL))

    B = steps[1:]/1e3
    cosalpha =  B[-1] / (2*R_earth)

    depths = R_earth - np.sqrt(B**2 + R_earth**2 - 2*B*R_earth*cosalpha)

    density = np.interp(depths, DPdepth, DPdensity, left=DPdensity[0], right=DPdensity[-1])
    density += rhoshift
    density *= rhoscale
    
    return oschelper.do_osc(E, steps, density, U, MP, dm21, dm31, nuind, dL)
