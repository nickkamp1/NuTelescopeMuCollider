import numpy as np
import matplotlib.pyplot as plt


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
lat = 41.88

# Baseline, approx sphere
alpha = ((90 - lat)/2)*np.pi/180
ICECUBE_BASELINE = np.sqrt(2*R_earth**2*(1 - np.cos((90+lat)*np.pi/180)))*1e3 # meters
# Others
PONE_BASELINE = 1758*1.60934*1e3
KM3NeT_BASELINE = 4427*1.60934*1e3

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



####################
# Helper Functions
####################

# for printing numbers in sci notation nicely
def sci_notation(number, sig_fig=1):
    ret_string_re = "{0:.{1:d}e}".format(np.real(number), sig_fig)
    ret_string_im = "{0:.{1:d}e}".format(np.imag(number), sig_fig)
    a_re, b_re = ret_string_re.split("e")
    a_im, b_im = ret_string_im.split("e")  
    # remove leading "+" and strip leading zeros
    b_re = int(b_re)
    b_im = int(b_im)
    print(a_re,b_re,a_im,b_im)
    if b_re!=0:
        a_im = "{0:.{1:d}f}".format(float(a_im)*10**(float(b_im)/b_re), sig_fig)
        b = b_re
    elif b_im!=0:
        a_re = "{0:.{1:d}f}".format(float(a_re)*10**(float(b_re)/b_im), sig_fig)
        b = b_im
    if float(a_re)==0: a_re="0"
    if float(a_im)==0: a_im="0"
    return r"(%s + %si)"%(a_re,a_im) + r" \times 10^{%s}"%str(b)

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


#####################
# Plotting functions
#####################

def energy_plot_1D(Enu,Emuon,
                   experiment_list,
                   rates_per_energy_SM,
                   rates_per_energy_LV,
                   nulabel,
                   aeu=0, aet = 0, aut = 0, # d = 3
                   ceu=0, cet=0, cut=0, # d = 4
                  ):
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot([], [],color="black",label="SM Only")
    ax[0].plot([], [],color="black",ls="--",label="SM + LV")
    for experiment,color in zip(experiment_list,["tab:green","tab:blue","tab:orange"]):
        ax[0].plot(Enu/1e3,1e3*rates_per_energy_SM[experiment],color=color,label=experiment)
        ax[0].plot(Enu/1e3,1e3*rates_per_energy_LV[experiment],color=color,ls="--")
        ax[1].plot(Enu/1e3,rates_per_energy_LV[experiment]/rates_per_energy_SM[experiment],color=color)
        
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    ax[1].set_xlabel("Neutrino Energy [TeV]")
    ax[0].set_ylabel("$%s$ Interactions / TeV / $5 \\times 10^{15}\\, \\mu$ Decays"%nulabel)
    ax[1].set_ylim(0.5,1.5)
    ax[0].set_xlim(Enu[0]/1e3,Enu[-1]/1e3)
    ax[1].set_xlim(Enu[0]/1e3,Enu[-1]/1e3)
    
    ax[0].text(0.01, 0.98, "%i TeV $\\mu^+$ Beam\nUnpolarized\n\n%i Mt Fid. Mass" % (Emuon/1e3, MASS/1e6), 
             transform=ax[0].transAxes, verticalalignment="top", fontsize=13)
    
    
    ax[0].legend(fontsize=12, columnspacing=1, frameon=False, ncol=3,
              loc='upper center', bbox_to_anchor=(0.45, 1.175))
    
    text_heights = np.linspace(1e7,1e8,3)
    text_x_a,text_x_c = 0.510,3
    if np.abs(aeu)>0: ax[0].text(text_x_a,text_heights[2],r"$a^{\rm T}_{e \mu} = %s~{\rm GeV}^{-1}$"%(sci_notation(aeu/1e9) if np.abs(aeu)>0 else "0"))
    if np.abs(aet)>0: ax[0].text(text_x_a,text_heights[1],r"$a^{\rm T}_{e \tau} = %s~{\rm GeV}^{-1}$"%(sci_notation(aet/1e9) if np.abs(aet)>0 else "0"))
    if np.abs(aut)>0: ax[0].text(text_x_a,text_heights[0],r"$a^{\rm T}_{\mu \tau} = %s~{\rm GeV}^{-1}$"%(sci_notation(aut/1e9) if np.abs(aut)>0 else "0"))
    if np.abs(ceu)>0: ax[0].text(text_x_c,text_heights[2],r"$c^{\rm TT}_{e \mu} = %s$"%(sci_notation(ceu) if np.abs(ceu)>0 else "0"))
    if np.abs(cet)>0: ax[0].text(text_x_c,text_heights[1],r"$c^{\rm TT}_{e \tau} = %s$"%(sci_notation(cet) if np.abs(cet)>0 else "0"))
    if np.abs(cut)>0: ax[0].text(text_x_c,text_heights[0],r"$c^{\rm TT}_{\mu \tau} = %s$"%(sci_notation(cut) if np.abs(cut)>0 else "0"))
    plt.show()


def radial_plot_1D(Rs,Emuon,
                   experiment_list,
                   rates_per_radius_SM,
                   rates_per_radius_LV,
                   nutype, nulabel,
                   aeu=0, aet = 0, aut = 0, # d = 3
                   ceu=0, cet=0, cut=0, # d = 4
                  ):
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot([], [],color="black",label="SM Only")
    ax[0].plot([], [],color="black",ls="--",label="SM + LV")
    for experiment,color in zip(experiment_list,["tab:green","tab:blue","tab:orange"]):
        ax[0].plot(Rs,rates_per_radius_SM[(nutype,experiment)],color=color,label=experiment)
        ax[0].plot(Rs,rates_per_radius_LV[(nutype,experiment)],color=color,ls="--")
        ax[1].plot(Rs,rates_per_radius_LV[(nutype,experiment)]/rates_per_radius_SM[(nutype,experiment)],color=color)
        
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    ax[1].set_xlabel("Neutrino Interaction Radius [m]")
    ax[0].set_ylabel("$%s$ Interactions / m / $5 \\times 10^{15}\\, \\mu$ Decays"%nulabel)
    ax[1].set_ylim(0.5,1.5)
    ax[0].set_xlim(Rs[0],Rs[-1])
    ax[1].set_xlim(Rs[0],Rs[-1])
    
    ax[0].text(0.68, 0.98, "%i TeV $\\mu^+$ Beam\nUnpolarized\n\n%i Mt Fid. Mass" % (Emuon/1e3, MASS/1e6), 
             transform=ax[0].transAxes, verticalalignment="top", fontsize=13)
    
    
    ax[0].legend(fontsize=12, columnspacing=1, frameon=False, ncol=3,
              loc='upper center', bbox_to_anchor=(0.45, 1.175))
    
    text_heights = np.linspace(2e7,3e7,3)
    text_x_a,text_x_c = 200,300
    if np.abs(aeu)>0: ax[0].text(text_x_a,text_heights[2],r"$a^{\rm T}_{e \mu} = %s~{\rm GeV}^{-1}$"%(sci_notation(aeu/1e9) if np.abs(aeu)>0 else "0"))
    if np.abs(aet)>0: ax[0].text(text_x_a,text_heights[1],r"$a^{\rm T}_{e \tau} = %s~{\rm GeV}^{-1}$"%(sci_notation(aet/1e9) if np.abs(aet)>0 else "0"))
    if np.abs(aut)>0: ax[0].text(text_x_a,text_heights[0],r"$a^{\rm T}_{\mu \tau} = %s~{\rm GeV}^{-1}$"%(sci_notation(aut/1e9) if np.abs(aut)>0 else "0"))
    if np.abs(ceu)>0: ax[0].text(text_x_c,text_heights[2],r"$c^{\rm TT}_{e \mu} = %s$"%(sci_notation(ceu) if np.abs(ceu)>0 else "0"))
    if np.abs(cet)>0: ax[0].text(text_x_c,text_heights[1],r"$c^{\rm TT}_{e \tau} = %s$"%(sci_notation(cet) if np.abs(cet)>0 else "0"))
    if np.abs(cut)>0: ax[0].text(text_x_c,text_heights[0],r"$c^{\rm TT}_{\mu \tau} = %s$"%(sci_notation(cut) if np.abs(cut)>0 else "0"))
    plt.show()
                   


####################
# Main Class
####################

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
            dL = baseline/1000

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
                                    beta : str,
                                    R = 0) -> list:
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

        baseline = np.sqrt(R**2 + self.baseline**2)

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
                    Re_term += mixing_element.real * (np.sin(baseline*deltaE / (2*hbarc*1e-2)))**2
                    Im_term += mixing_element.imag * (np.sin(baseline*deltaE / (hbarc*1e-2)))**2
            P -= 4*Re_term
            P += 2*Im_term
            Plist.append(P)
        return np.array(Plist)



                   
