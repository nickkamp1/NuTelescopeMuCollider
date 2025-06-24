from scipy.stats import norm
import numpy as np
from global_variables import *

def NueEnergySmearing(Ereco,Etrue,kE=0.1/np.log10(np.e)):
    return norm.pdf(Ereco,loc=Etrue,scale=kE*Etrue)

def NumuEnergySmearing(Ereco,Etrue,kE=0.5):
    return norm.pdf(Ereco,loc=Etrue,scale=kE*Etrue)

def PositionSmearing(Rreco,Rtrue,sig_r=10):
    return norm.pdf(Rreco,loc=Rtrue,scale=sig_r)

# this function must consider the nue CC rates as well as the numu and nue NC rates
def cascade_integrand(x,
                      nue_interp_rates_per_energy_radius_SM,
                      nue_interp_rates_per_energy_radius_alt):
    Er,Rr,Et = x
    Rt = Rr
    #Et = Er
    ret = {}
    for experiment in experiment_list:
        ret[(experiment,"SM")] = nue_interp_rates_per_energy_radius_SM[experiment]((Et,Rt))*NueEnergySmearing(Er,Et)#*PositionSmearing(Rr,Rt)
        ret[(experiment,"alt")] = nue_interp_rates_per_energy_radius_alt[experiment]((Et,Rt))*NueEnergySmearing(Er,Et)#*PositionSmearing(Rr,Rt)
    return ret

# this function need only consider numu CC rates
def track_integrand(x,
                      numubar_interp_rates_per_energy_radius_SM,
                      numubar_interp_rates_per_energy_radius_alt):
    Et,Rr = x
    Rt = Rr
    ret = {}
    for experiment in experiment_list:
        ret[(experiment,"SM")] = numubar_interp_rates_per_energy_radius_SM[experiment]((Et,Rt))#*PositionSmearing(Rr,Rt)
        ret[(experiment,"alt")] = numubar_interp_rates_per_energy_radius_alt[experiment]((Et,Rt))#*PositionSmearing(Rr,Rt)
    return ret

def trapezoid_integrate(Rbins,Ebins,
                        nue_interp_rates_per_energy_radius_SM,
                        nue_interp_rates_per_energy_radius_alt,
                        numubar_interp_rates_per_energy_radius_SM,
                        numubar_interp_rates_per_energy_radius_alt,
                        oversampling=10):
    numubar_rates_per_bin = {(exp,k):np.zeros(len(Rbins)-1) for exp in experiment_list for k in ["SM","alt"]} # radial bins 
    nue_rates_per_bin = {(exp,k):np.zeros((len(Ebins)-1,len(Rbins)-1)) for exp in experiment_list for k in ["SM","alt"]} # energy x radial bins
    for iR in range(len(Rbins)-1):
        r_subrange = np.linspace(Rbins[iR],Rbins[iR+1],oversampling)
        e_subrange = np.linspace(Ebins[0],Ebins[-1],len(Ebins)*oversampling) # integerate numu over full e range
        X,Y = np.meshgrid(e_subrange,r_subrange)
        numubar_diff_rates = track_integrand((X,Y),
                                               numubar_interp_rates_per_energy_radius_SM,
                                               numubar_interp_rates_per_energy_radius_alt)
        for exp in experiment_list:
            for k in ["SM","alt"]:
                numubar_rate = np.trapz(np.trapz(numubar_diff_rates[(exp,k)], r_subrange, axis=0), e_subrange, axis=0)
                numubar_rates_per_bin[(exp,k)][iR] = numubar_rate
        for iE in range(len(Ebins)-1):
            e_subrange = np.linspace(Ebins[iE],Ebins[iE+1],oversampling) # integerate nue over only e bin
            etrue_subrange = np.linspace(Ebins[0],Ebins[-1],len(Ebins)*oversampling) # integrate over etrue as well for energy smearing
            X,Y,Z = np.meshgrid(e_subrange,r_subrange,etrue_subrange)
            nue_diff_rates = cascade_integrand((X,Y,Z),
                                           nue_interp_rates_per_energy_radius_SM,
                                           nue_interp_rates_per_energy_radius_alt)
            for exp in experiment_list:
                for k in ["SM","alt"]:
                    nue_rate = np.trapz(np.trapz(np.trapz(nue_diff_rates[(exp,k)], e_subrange, axis=0), r_subrange, axis=0), etrue_subrange, axis=0)
                    nue_rates_per_bin[(exp,k)][iE,iR] = nue_rate
    return numubar_rates_per_bin,nue_rates_per_bin

def best_fit_N(nue_alt_rates_per_bin,
               nue_SM_rates_per_bin,
               numubar_alt_rates_per_bin,
               numubar_SM_rates_per_bin,
               sigmaN=0.01):
    SM_tot = np.sum(nue_SM_rates_per_bin) + np.sum(numu_SM_rates_per_bin)
    alt_tot = np.sum(nue_alt_rates_per_bin) + np.sum(numu_alt_rates_per_bin)
    a = 1/(sigmaN**2)
    b = SM_tot - 1/(sigmaN**2)
    c = -alt_tot
    return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

def asimov_discovery_sensitivity(nue_alt_rates_per_bin,
                                 nue_SM_rates_per_bin,
                                 numu_alt_rates_per_bin,
                                 numu_SM_rates_per_bin,
                                 sigmaN=0.01):
    N = best_fit_N(nue_alt_rates_per_bin,
                   nue_SM_rates_per_bin,
                   numu_alt_rates_per_bin,
                   numu_SM_rates_per_bin,
                   sigmaN=sigmaN)
    nue_deltaLLH = nue_alt_rates_per_bin*(np.log(nue_alt_rates_per_bin) - np.log(N*nue_SM_rates_per_bin)) - (nue_alt_rates_per_bin - N*nue_SM_rates_per_bin)
    nue_deltaLLH = np.sum(np.where(np.isnan(nue_deltaLLH),0,nue_deltaLLH))
    numu_deltaLLH = numu_alt_rates_per_bin*(np.log(numu_alt_rates_per_bin) - np.log(N*numu_SM_rates_per_bin)) - (numu_alt_rates_per_bin - N*numu_SM_rates_per_bin)
    numu_deltaLLH = np.sum(np.where(np.isnan(numu_deltaLLH),0,numu_deltaLLH))
    norm_deltaLLH = (N-1)**2 / (2*sigmaN**2)
    return 2*(nue_deltaLLH + numu_deltaLLH + norm_deltaLLH)

def delta_chi_square(nue_alt_rates_per_bin,
                     nue_SM_rates_per_bin,
                     numu_alt_rates_per_bin,
                     numu_SM_rates_per_bin,
                     sigmaCorr=0.01,
                     sigmaUncorr=0,
                     threshold=1):
    
    predSM = np.concatenate((nue_SM_rates_per_bin.flatten(),numu_SM_rates_per_bin))
    predalt = np.concatenate((nue_alt_rates_per_bin.flatten(),numu_alt_rates_per_bin))
    nonzero_mask = np.where(predSM>threshold)
    predSM = predSM[nonzero_mask]
    predalt = predalt[nonzero_mask]
    delta = (predalt - predSM)
    fracCov = (sigmaCorr**2)*np.ones((len(predSM),len(predSM)))
    fracCov += np.diag(sigmaUncorr**2 + 1/predSM)
    Cov = fracCov * np.outer(predSM,predSM)
    deltaChiSquare = np.matmul(delta.T,np.matmul(np.linalg.inv(Cov),delta))
    return deltaChiSquare
