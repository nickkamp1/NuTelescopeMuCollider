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
    return {exp:delta_chi_square(nue_rates_per_bin[(exp,"alt")],
                                 nue_rates_per_bin[(exp,"SM")],
                                 numubar_rates_per_bin[(exp,"alt")],
                                 numubar_rates_per_bin[(exp,"SM")],
                                 sigmaCorr=sigmaCorr,
                                 sigmaUncorr=sigmaUncorr) for exp in experiment_list}