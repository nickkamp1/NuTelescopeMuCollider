import numpy as np

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

deltaCP0 = 0.

# function to get the average electron density along the neutrino line of sight
# baseline is the distance of the beam, and dL is the step size with which to compute the average density
# returns average density in CGI units (g / cm^3)
def get_average_electron_density(baseline, dL, rhoshift, rhoscale):
    steps = np.linspace(0, baseline, int(baseline/dL))

    B = steps[1:]/1e3
    cosalpha =  B[-1] / (2*R_earth)

    depths = R_earth - np.sqrt(B**2 + R_earth**2 - 2*B*R_earth*cosalpha)

    density = np.interp(depths, DPdepth, DPdensity, left=DPdensity[0], right=DPdensity[-1])
    density += rhoshift
    density *= rhoscale
    return np.average(density) * nucleon_mass_density_to_electron_number_density

# This class is designed to handle the oscillation probability given SM extension coefficients
# follows appendix A of 1410.4267
# note that we do not time-evolve the neutrino wavefunction
# rather, we take the average electron density along the neutrino path
# and compute eq A17 to get the oscillation probability

class LV_oscillations:

    def __init__(self, energies, baseline, nusign, dL=None, # m
                 rhoset=None, rhoshift=0, rhoscale=1,
                 # three-neutrino oscillations
                 th12=th12, th23=th23, th13=th13, dm21=dm21, dm31=dm31,
                 # lorentz violation parameters
                 aeu=0, aet = 0, aut = 0, # d = 3
                 ceu=0, cet=0, cut=0, # d = 4
                 # anomalous matter potential
                 epsee=0, epsemu=0, epsetau=0, epsmumu=0, epsmutau=0, epstautau=0):

        if isinstance(energies, float) or isinstance(energies, int):
            energies = np.array([energies]).astype(float)
        self.energies = energies
        if dL is None:
            dL = baseline/100

        c12 = np.cos(th12)
        c23 = np.cos(th23)
        c13 = np.cos(th13)

        s12 = np.sin(th12)
        s23 = np.sin(th23)
        s13 = np.sin(th13)

        # 3 neutrino vacuum oscillations -- from https://pdg.lbl.gov/2020/reviews/rpp2020-rev-neutrino-mixing.pdf
        U = np.array([
        [c12*c13,	c13*s12,	s13],
        [-c23*s12 -c12*s13*s23,	c12*c23 -s12*s13*s23,	c13*s23],
        [-c12*c23*s13 + s12*s23, -c23*s12*s13 -c12*s23,	c13*c23]
        ])

        if nusign < 0:
            U = np.conj(U)

        U_dagger = U.conj().T

        Vlist = [] # list for each energy considered
        for energy in energies:
            M = np.array([
                [0, 0, 0],
                [0, dm21/(2*energy), 0],
                [0, 0, dm31 / (2*energy)]
            ])*GeV
            Vlist.append(np.matmul(np.matmul(U, M), U_dagger))


        # Matter Potential
        self.baseline = baseline
        Ne_avg = get_average_electron_density(baseline, dL, rhoshift, rhoscale)
        MP = nusign*np.sqrt(2)*GFermi*Ne_avg*np.array([
            [1-0.5+epsee, epsemu, epsetau],
            [np.conj(epsemu), -0.5+epsmumu, epsmutau],
            [np.conj(epsetau), np.conj(epsmutau), -0.5+epstautau]
        ])*GeV

        # a terms
        A = nusign*np.array([
            [0, aeu, aet],
            [np.conj(aeu), 0, aut],
            [np.conj(aet), np.conj(aut), 0]
        ])*GeV

        if nusign < 0:
            A = np.conj(A)

        # c terms
        Clist = [] # list for each energy considered
        for energy in energies:
            C = -4./3.*energy*np.array([
                [0, ceu, cet],
                [np.conj(ceu), 0, cut],
                [np.conj(cet), np.conj(cut), 0]
                  ])*GeV
            if nusign > 0:
                Clist.append(C)
            else:
                Clist.append(np.conj(C))

        # lists for each energy considered
        # these will be used to compute oscillation probabilities later on
        self.E = {0:[],1:[],2:[]}
        self.Ue = {0:[],1:[],2:[]}
        self.Umu = {0:[],1:[],2:[]}
        self.Utau = {0:[],1:[],2:[]}
        for V,C in zip(Vlist,Clist):
            H = V + MP + A + C # total hamiltonian
            a = -np.trace(H)
            b = (a**2 - np.trace(np.matmul(H, H)))/2
            c = -np.linalg.det(H)
            R = (2*(a**3) - 9*a*b + 27*c) / 54
            Q = (a**2 - 3*b)/9
            theta0 = np.arccos(R*(Q**(-3./2.)))
            theta = {0:theta0,
                     1:theta0+2*np.pi,
                     2:theta0-2*np.pi}
            for i in range(3):
                Ei = -2*np.sqrt(Q)*np.cos(theta[i]/3) - a/3
                self.E[i].append(Ei)
                Ai = H[1,2]*(H[0,0] - Ei) - H[1,0]*H[0,2]
                Bi = H[2,0]*(H[1,1] - Ei) - H[2,1]*H[1,0]
                Ci = H[1,0]*(H[2,2] - Ei) - H[1,2]*H[2,0]
                AB = Ai*Bi
                AC = Ai*Ci
                BC = Bi*Ci
                Ni = np.sqrt(np.conj(AB)*AB + np.conj(AC)*AC + np.conj(BC)*BC)
                self.Ue[i].append(np.conj(Bi)*Ci/Ni)
                self.Umu[i].append(Ai*Ci/Ni)
                self.Utau[i].append(Ai*Bi/Ni)

    # alpha = initial flavor
    # beta = final flavor
    # returns oscillation probability for alpha -> beta at each energy
    def get_oscillation_probability(self,
                                    alpha : str,
                                    beta : str) -> list:
        if alpha not in ['e', 'mu', 'tau'] or beta not in ['e', 'mu', 'tau']:
            raise ValueError("Invalid initial flavor. Must be 'e', 'mu', or 'tau'")

        # set the alpha flavor
        if alpha == "e": Ualpha = self.Ue
        elif alpha == "mu": Ualpha = self.Umu
        elif alpha == "tau": Ualpha = self.Utau

        # set the beta flavor
        if beta == "e": Ubeta = self.Ue
        elif beta == "mu": Ubeta = self.Umu
        elif beta == "tau": Ubeta = self.Utau

        # compute the oscillation probability for each energy
        Plist = []
        for energy_idx,energy in enumerate(self.energies):
            P = alpha==beta # delta function for app/disapp
            Re_term = 0
            Im_term = 0
            for i in range(3):
                for j in range(3):
                    if i <= j: continue
                    deltaE = self.E[j][energy_idx] - self.E[i][energy_idx]
                    mixing_element = Ubeta[j][energy_idx] * np.conj(Ubeta[i][energy_idx]) * np.conj(Ualpha[j][energy_idx]) * Ualpha[i][energy_idx]
                    Re_term += mixing_element.real * (np.sin(self.baseline*deltaE / (2*hbarc*1e-2)))**2
                    Im_term += mixing_element.imag * (np.sin(self.baseline*deltaE / (hbarc*1e-2)))**2
            P -= 4*Re_term
            P += 2*Im_term
            Plist.append(P)
        return np.array(Plist)
