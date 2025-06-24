from scipy.interpolate import RegularGridInterpolator

import osc

from binned_analysis import *
from global_variables import *
from flux import *
from xsec import *

import os


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
                                       Enus = np.linspace(Emuon/10, Emuon, 500),
                                       Rs = np.linspace(0,RADIUS,100),
                                       dm41=0, th42=0, norm_scale=1):

    dE = Enus[1] - Enus[0]
    baseline = baseline_list[experiment]
    
    numubar_rates_per_energy_radius = np.zeros((len(Enus),len(Rs)))
    nue_rates_per_energy_radius = np.zeros((len(Enus),len(Rs)))

    numubarSM_rates_per_energy_radius = np.zeros((len(Enus),len(Rs)))
    nueSM_rates_per_energy_radius = np.zeros((len(Enus),len(Rs)))

    numubar_osc = np.abs(osc.oscillate(Enus, baseline, 1, -1, dL=10, dm41=dm41, th24=th42))**2
    numubar_osc = numubar_osc / np.sum(numubar_osc, axis=0) # force unitarity, fix numerical error

    numubar_SM = np.abs(osc.oscillate(Enus, baseline, 1, -1, dL=10))**2
    numubar_SM = numubar_SM / np.sum(numubar_SM, axis=0) # force unitarity, fix numerical error
    
    # fix rapid oscillations
    rapid = (dm41*1.27*baseline*dE / 1e3 / Enus**2 / (2*np.pi)) > 0.05 # tuned cut
    numubar_osc[:, rapid] = numubar_SM[:, rapid]*(1 - np.sin(2*th42)**2/2)

    for ir,R in enumerate(Rs):
        mu_to_mu = numubar_osc[1]
        mu_to_mu_SM = numubar_SM[1]
        for ie,E in enumerate(Enus):
            numubar_rates_per_energy_radius[ie,ir] = mu_to_mu[ie]*numubar_rate(E, R, P, baseline)*norm_scale
            nue_rates_per_energy_radius[ie,ir] = 0

            numubarSM_rates_per_energy_radius[ie,ir] = mu_to_mu_SM[ie]*numubar_rate(E, R, P, baseline)*norm_scale
            nueSM_rates_per_energy_radius[ie,ir] = 0

    # interpolation function
    numubar_interp_rates_per_energy_radius = RegularGridInterpolator((Enus,Rs),numubar_rates_per_energy_radius)
    nue_interp_rates_per_energy_radius = RegularGridInterpolator((Enus,Rs),nue_rates_per_energy_radius)

    numubarSM_interp_rates_per_energy_radius = RegularGridInterpolator((Enus,Rs),numubarSM_rates_per_energy_radius)
    nueSM_interp_rates_per_energy_radius = RegularGridInterpolator((Enus,Rs),nueSM_rates_per_energy_radius)
    
    return (numubarSM_interp_rates_per_energy_radius, 
            nueSM_interp_rates_per_energy_radius, 
            numubar_interp_rates_per_energy_radius, 
            nue_interp_rates_per_energy_radius)

def get_delta_chisquare_sterilenu(Rbins,Ebins,
                                    dm41=0, th42=0,
                                    sigmaCorr=0.01, sigmaUncorr=0,
                                 norm_scale=1.):

    nue_interp_rates_per_energy_radius_SM = {}
    nue_interp_rates_per_energy_radius_LV = {}
    numubar_interp_rates_per_energy_radius_SM = {}
    numubar_interp_rates_per_energy_radius_LV = {}    

    # check if all cache files exist
    has_all_cache = True
    for exp in experiment_list:
        for var in ["alt", "SM"]:
            if not os.path.isfile("rates/dm%f_th%f_%s_%s_numubar_rates.txt" % (dm41, th42, exp, var)):
                has_all_cache = False
                break
        if not has_all_cache:
            break

    if has_all_cache:
        numubar_rates_per_bin = {}
        nue_rates_per_bin = {}
        for exp in experiment_list:
            for var in ["alt", "SM"]:
                numubar_rates_per_bin[(exp, var)] = np.loadtxt("rates/dm%f_th%f_%s_%s_numubar_rates.txt" % (dm41, th42, exp, var))*norm_scale
                nue_rates_per_bin[(exp, var)] = np.zeros((Rbins.size-1, Ebins.size-1))
                

    else:
        for exp in experiment_list:
            (numubar_interp_rates_per_energy_radius_SM[exp],
             nue_interp_rates_per_energy_radius_SM[exp], 
             numubar_interp_rates_per_energy_radius_LV[exp],
             nue_interp_rates_per_energy_radius_LV[exp]) = get_interp_rates_per_energy_radius(exp, dm41=dm41, th42=th42, norm_scale=norm_scale)
            
        numubar_rates_per_bin,nue_rates_per_bin = trapezoid_integrate(Rbins,Ebins,
                                                                   nue_interp_rates_per_energy_radius_SM,
                                                                   nue_interp_rates_per_energy_radius_LV,
                                                                   numubar_interp_rates_per_energy_radius_SM,
                                                                   numubar_interp_rates_per_energy_radius_LV)
    
        for exp in experiment_list:
            np.savetxt("rates/dm%f_th%f_%s_%s_numubar_rates.txt" % (dm41, th42, exp, "alt"), numubar_rates_per_bin[(exp,"alt")])
            np.savetxt("rates/dm%f_th%f_%s_%s_numubar_rates.txt" % (dm41, th42, exp, "SM"), numubar_rates_per_bin[(exp,"SM")])

    return {exp:delta_chi_square(nue_rates_per_bin[(exp,"alt")],
                                 nue_rates_per_bin[(exp,"SM")],
                                 numubar_rates_per_bin[(exp,"alt")],
                                 numubar_rates_per_bin[(exp,"SM")],
                                 sigmaCorr=sigmaCorr,
                                 sigmaUncorr=sigmaUncorr) for exp in experiment_list}

