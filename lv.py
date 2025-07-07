import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from flux import *
from xsec import *
from binned_analysis import *
from global_variables import *


####################
# Helper Functions
####################

#####################
# Rate functions
#####################

# Rate of neutrino interactions / dR / dE
# Units: (Number of Interactions) / m / GeV
def rate(flux, xsec, Enu, R, P, baseline):
    costh = np.cos(R/baseline)
    ret = flux(Emuon, P, Enu, costh, baseline)*xsec(Enu)*MASS*(4*np.pi*R*np.sqrt(RADIUS**2 - R**2)/VOLUME)
    if not isinstance(ret, float):
        ret[R > RADIUS] = 0
    elif R > RADIUS:
        return 0
    return ret

def numu_rate(Enu, R, P, B):
    return rate(numu_flux_baseline, xsec, Enu, R, P, B)

def nue_rate(Enu, R, P, B):
    return rate(nue_flux_baseline, xsec, Enu, R, P, B)

def numubar_rate(Enu, R, P, B):
    return rate(numu_flux_baseline, xsecbar, Enu, R, P, B)

def nuebar_rate(Enu, R, P, B):
    return rate(nue_flux_baseline, xsecbar, Enu, R, P, B)

# function to get the interpolation functions of numubar and nue rate for a given experiment
# per energy and radius, within the bounds of Enus and Rs
def get_interp_rates_per_energy_radius(experiment,
                                       Enus = np.linspace(Emuon/100, Emuon, 100),
                                       Rs = np.linspace(0,RADIUS,100),
                                       aeu = 0, aet = 0, aut = 0,
                                       ceu = 0, cet = 0, cut = 0):

    baseline = baseline_list[experiment]
    LV_case_nu = LV_oscillations(Enus*1e9, baseline, 1,
                                 aeu = aeu, aet = aet, aut = aut,
                                 ceu = ceu, cet = cet, cut = cut)
    LV_case_nubar = LV_oscillations(Enus*1e9, baseline, -1,
                                    aeu = aeu, aet = aet, aut = aut,
                                    ceu = ceu, cet = cet, cut = cut)
    
    numubar_rates_per_energy_radius = np.zeros((len(Enus),len(Rs)))
    nue_rates_per_energy_radius = np.zeros((len(Enus),len(Rs)))

    for ir,R in enumerate(Rs):
        # numubars
        mu_to_mu = LV_case_nubar.get_oscillation_probability("mu", "mu",R=R)
        e_to_mu = LV_case_nubar.get_oscillation_probability("e", "mu",R=R)
        # nues
        e_to_e = LV_case_nubar.get_oscillation_probability("e", "e",R=R)
        mu_to_e = LV_case_nubar.get_oscillation_probability("mu", "e",R=R)
        for ie,E in enumerate(Enus):
            numubar_rates_per_energy_radius[ie,ir] = mu_to_mu[ie] * numubar_rate(E, R, P, baseline) + e_to_mu[ie] * nue_rate(E, R, P, baseline)
            nue_rates_per_energy_radius[ie,ir] = e_to_e[ie] * nue_rate(E, R, P, baseline) + mu_to_e[ie] * numubar_rate(E, R, P, baseline)
    # interpolation functions
    numubar_interp_rates_per_energy_radius = RegularGridInterpolator((Enus,Rs),numubar_rates_per_energy_radius)
    nue_interp_rates_per_energy_radius = RegularGridInterpolator((Enus,Rs),nue_rates_per_energy_radius)
    return numubar_interp_rates_per_energy_radius,nue_interp_rates_per_energy_radius
    
        
def get_delta_chisquare_for_LV_case(Rbins,Ebins,
                                    aeu = 0, aet = 0, aut = 0,
                                    ceu = 0, cet = 0, cut = 0,
                                    sigmaCorr=0.015,sigmaUncorr=0):

    nue_interp_rates_per_energy_radius_SM = {}
    nue_interp_rates_per_energy_radius_LV = {}
    numubar_interp_rates_per_energy_radius_SM = {}
    numubar_interp_rates_per_energy_radius_LV = {}
    for exp in experiment_list:
        (numubar_interp_rates_per_energy_radius_SM[exp],
         nue_interp_rates_per_energy_radius_SM[exp]) = get_interp_rates_per_energy_radius(exp,aeu = 0, aet = 0, aut = 0,
                                                                                        ceu = 0, cet = 0, cut = 0)
        (numubar_interp_rates_per_energy_radius_LV[exp],
         nue_interp_rates_per_energy_radius_LV[exp]) = get_interp_rates_per_energy_radius(exp,aeu = aeu, aet = aet, aut = aut,
                                                                                        ceu = ceu, cet = cet, cut = cut)
    numubar_rates_per_bin,nue_rates_per_bin = trapezoid_integrate(Rbins,Ebins,
                                                               nue_interp_rates_per_energy_radius_SM,
                                                               nue_interp_rates_per_energy_radius_LV,
                                                               numubar_interp_rates_per_energy_radius_SM,
                                                               numubar_interp_rates_per_energy_radius_LV)
    print("numubar total rate",sum(numubar_rates_per_bin[(exp,"SM")]))
    return {exp:delta_chi_square(nue_rates_per_bin[(exp,"alt")],
                                 nue_rates_per_bin[(exp,"SM")],
                                 numubar_rates_per_bin[(exp,"alt")],
                                 numubar_rates_per_bin[(exp,"SM")],
                                 sigmaCorr=sigmaCorr,
                                 sigmaUncorr=sigmaUncorr) for exp in experiment_list}
    


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
              loc='best', bbox_to_anchor=(0.45, 1.175))
    
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
              loc='best', bbox_to_anchor=(0.45, 1.175))
    
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
# Main Classses
####################


####################
# Isotropic Oscillations
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

####################
# Sidereal Variations
####################

def lat_long_to_cartesian(lat,long):
    x = R_earth*np.cos(lat)*np.cos(long)
    y = R_earth*np.cos(lat)*np.sin(long)
    z = R_earth*np.sin(lat)
    return np.array([x,y,z]).T
    
def directional_vector(experiment):
    lat = latitude_list[experiment]
    long = longitude_list[experiment]
    start = lat_long_to_cartesian(fermi_lat,fermi_long)
    end = lat_long_to_cartesian(lat,long)
    direction = end - start
    theta = np.arccos(np.dot(start,direction)/(np.linalg.norm(start)*np.linalg.norm(direction)))
    e = np.array([-np.sin(fermi_long),np.cos(fermi_long),0]) # east vector at beam
    n = np.array([-np.cos(fermi_lat)*np.cos(fermi_long), -np.cos(lat)*np.sin(fermi_long), np.sin(fermi_lat)]) # north vector at beam
    vxy = direction - (np.dot(start,direction)/np.dot(start,start))*start
    cosphi = np.dot(vxy,-n)/np.linalg.norm(vxy)
    sinphi = np.dot(vxy,e)/np.linalg.norm(vxy)
    chi = np.pi/180 * (90 - lat) # colatitude
    Nx = np.cos(chi)*np.sin(theta)*cosphi + np.sin(chi)*np.cos(theta)
    Ny = np.sin(theta)*sinphi
    Nz = -np.sin(chi)*np.sin(theta)*cosphi + np.cos(chi)*np.cos(theta)
    return np.array([Nx,Ny,Nz])

def osc_prob(L,RA,
             C=0,
             Ac=0,As=0,
             Bc=0,Bs=0,
             Emean=Emuon/2):
    L_term = (L/(hbarc*1e-2*1e-9))**2 # GeV^-2
    sm_term = (dm31*(1e-9)**2)/(4*Emean)*np.sin(2*th23)**2 # GeV
    lv_term = (C 
               + Ac*np.cos(2*np.pi*RA)
               + As*np.sin(2*np.pi*RA)
               + Bc*np.cos(2*2*np.pi*RA)
               + Bs*np.sin(2*2*np.pi*RA)
              ) # GeV
    #print(sm_term,lv_term)
    osc_expr = L_term * (sm_term + lv_term)**2
    if (np.any(osc_expr>1)):
        print("Warning! We are not in the short baseline regime. perturbation term reaches %1.1f"%np.max(osc_expr))
    return 1 - osc_expr

# This class is designed to handle sidereal variations given SM extension coefficients
# follows https://arxiv.org/abs/hep-ph/0406255
# we drop the flavor indices; all oscillation probabilities are from generic flavor a to b

class LV_sidreal_variations:

    # aL = 4-vector
    # cL = 4x4 tensor
    def __init__(self,
                 aL,
                 cL):
        self.aL = aL
        self.cL = cL

    # returns C, As, Ac, Bs, Bc
    # follows eq 4-12 of https://arxiv.org/abs/hep-ph/0406255
    def get_coefficients(self,
                         exp,
                         Emean=Emuon/2, # GeV
                         nusign=1):
        
        if nusign>0:
            aL = self.aL
            cL = self.cL
        else:
            aL = np.conj(self.aL)
            cL = np.conj(self.cL)

        N = directional_vector(exp)
        
        C0 = aL[0] - N[2]*aL[3]
        C1 = -1./2 * (3 - N[2]*N[2])*cL[0,0] + 2*N[2]*cL[0,2] + 1./2.*(1 - 3*N[2]*N[2])*cL[2,2]
        As0 = N[1]*aL[1] - N[0]*aL[2]
        As1 = -2*N[1]*cL[0,1] + 2*N[0]*cL[0,2] + 2*N[1]*N[2]*cL[1,3] - 2*N[0]*N[2]*cL[2,3]
        Ac0 = -N[0]*aL[1] - N[1]*aL[2]
        Ac1 = 2*N[0]*cL[0,1] + 2*N[1]*cL[0,2] - 2*N[0]*N[2]*cL[1,3] - 2*N[1]*N[2]*cL[2,3]
        Bs1 = N[1]*N[2] * (cL[1,1] - cL[2,2]) - (N[0]*N[0] - N[1]*N[1])*cL[1,2]
        Bc1 = -1./2.*(N[0]*N[0] - N[1]*N[1])*(cL[1,1] - cL[2,2]) - 2*N[0]*N[1]*cL[1,2]

        C = C0 + Emean*C1
        As = As0 + Emean*As1
        Ac = Ac0 + Emean*Ac1
        Bs = Emean*Bs1
        Bc = Emean*Bc1
        return C,As,Ac,Bs,Bc

    
        
        



                   
