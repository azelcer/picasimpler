import numpy as np
import pandas as pd
import logging as _lgn
import yaml
import json
import time as _time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial import distance
from scipy.optimize import curve_fit, fsolve
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
    
from picasimpler.helpers.status import AnalysisStatus, MessageType
from picasimpler.helpers.utils import px_to_nm
from picasimpler.helpers.cf_calc import SimulQ
from picasimpler.config.config_var import (
    MAX_FIRST_FRAME_PERC,
    MIN_LAST_FRAME_PERC,
    FRAME_MEDIAN_PERC_RANGE,
    MAX_ON_FRAMES_PERC,
    MIN_GOOD_LOC,
    RIFLE_N_CLUST_EXP,
    HB_N_CLUST_EXP,
    Z_BASELINE_NM,
    RIFLE_Z_SITES_NM,
    HB_Z_SITES_NM,
    DF_REF_VAL_NM,
    RES_DIR,
    ALPHA_GUESS,
    ALPHA_MAX,
    ALPHA_FIXED,
    D_GUESS,
    SPACER_GUESS,
    D_LONG_GUESS,
    CALIB_PLOT_RANGE_NM,
    CALIB_PLOT_PTS,
    Z_SIM_DISCR,
    Z_SIM_FIT_ARR,
    LAMBDA_EM_DISCR,
    CALIB_MODE
)

def fit_renorm_exp_appr(
    self,
    p0: tuple[float, float] = (ALPHA_GUESS, D_GUESS),
):
    """Calculate SIMPLER effective params from known z and N.

    This function fits N instead of z and should be used when N0 of each
    origami is unknown.

    In this case, the fitting function is an exponential + constant (which is an approximation)

    Parameters
    ----------
        p0: tuple[float, float], OPTIONAL
            Initial guesses for alpha_F and d_F

    TODO: add z_0 to account for constant linker added distance.
    """
    
    N_data = self.clust_means[:, :, 2]

    flat_z = self.z_real[:, 1:].ravel()
    # Normalize datal
    F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
    z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))

    # Model function
    def F(z, alpha_F, d_F):
        num = alpha_F * np.exp(-z / d_F) + (1 - alpha_F)
        den = alpha_F * np.exp(-z_0 / d_F) + (1 - alpha_F)
        return num / den
    # Fit
    popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0,0],[self.alpha_max, np.inf]))
    alpha_F, d_F = popt

    perr = np.sqrt(np.diag(pcov))
    self.alpha_F = alpha_F
    self.d_F = d_F
    self.alpha_F_err = perr[0]
    self.d_F_err = perr[1]
    
def fit_renorm_no_appr(
    self,
    p0: tuple[float, float] = (ALPHA_GUESS, D_GUESS),
):
    """Calculate SIMPLER effective params from known z and N.

    This function fits N instead of z and should be used when N0 of each
    origami is unknown.

    In this case, the fitting function is an exponential + constant multiplied by the CF

    Parameters
    ----------
        p0: tuple[float, float], OPTIONAL
            Initial guesses for alpha_F and d_F

    TODO: add z_0 to account for constant linker added distance.
    """
    self._calc_coll_fl_arr_axelrod()
    N_data = self.clust_means[:, :, 2]
    flat_z = self.z_real[:, 1:].ravel()
    # Normalize datal
    F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
    z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
    # Model function
    def F(z, alpha_exc, d_exc):
        num = (alpha_exc * np.exp(-(z) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
        den = (alpha_exc * np.exp(-(z_0) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
        return num / den
    # Fit
    popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0,0],[self.alpha_max, np.inf]))
    alpha_exc, d_exc = popt

    perr = np.sqrt(np.diag(pcov))
    self.alpha_exc = alpha_exc
    self.d_exc = d_exc
    self.alpha_exc_err = perr[0]
    self.d_exc_err = perr[1]
    
    tirf_angle_lambda_factor = self.params.lambda_exc / (4*np.pi)
    tirf_angle_sqrt_factor = np.sqrt(((tirf_angle_lambda_factor / self.d_exc)**2 + self.params.n_s**2))
    tirf_angle_sin = tirf_angle_sqrt_factor / self.params.n_i
    tirf_angle_deriv = (1 / np.sqrt(1 - tirf_angle_sin**2)) * (1 / self.params.n_i) * (1/2) * (1 / tirf_angle_sqrt_factor) * tirf_angle_lambda_factor**2 * (2/self.d_exc**3) * (180/np.pi)
    self.tirf_angle = np.arcsin(tirf_angle_sin)*180/np.pi
    self.tirf_angle_err = np.abs(tirf_angle_deriv) * self.d_exc_err
    
def fit_renorm_no_appr_spacer(
    self,
    p0: tuple[float, float] = (ALPHA_GUESS, D_GUESS, SPACER_GUESS),
):
    """Calculate SIMPLER effective params from known z and N.

    This function fits N instead of z and should be used when N0 of each
    origami is unknown.

    In this case, the fitting function is an exponential + constant multiplied by the CF

    Parameters
    ----------
        p0: tuple[float, float], OPTIONAL
            Initial guesses for alpha_F and d_F

    TODO: add z_0 to account for constant linker added distance.
    """
    self._calc_coll_fl_arr_axelrod()
    N_data = self.clust_means[:, :, 2]
    flat_z = self.z_real[:, 1:].ravel()
    # Normalize datal
    F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
    z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
    # Model function
    def F(z, alpha_exc, d_exc, spacer):
        num = (alpha_exc * np.exp(-(z + spacer) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z + spacer)
        den = (alpha_exc * np.exp(-(z_0 + spacer) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0 + spacer)
        return num / den
    # Fit
    popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0,0,0],[self.alpha_max, np.inf, 0.1]))
    alpha_exc, d_exc, spacer = popt

    perr = np.sqrt(np.diag(pcov))
    self.alpha_exc = alpha_exc
    self.d_exc = d_exc
    self.alpha_exc_err = perr[0]
    self.d_exc_err = perr[1]
    
    print(spacer)
    
    tirf_angle_lambda_factor = self.params.lambda_exc / (4*np.pi)
    tirf_angle_sqrt_factor = np.sqrt(((tirf_angle_lambda_factor / self.d_exc)**2 + self.params.n_s**2))
    tirf_angle_sin = tirf_angle_sqrt_factor / self.params.n_i
    tirf_angle_deriv = (1 / np.sqrt(1 - tirf_angle_sin**2)) * (1 / self.params.n_i) * (1/2) * (1 / tirf_angle_sqrt_factor) * tirf_angle_lambda_factor**2 * (2/self.d_exc**3) * (180/np.pi)
    self.tirf_angle = np.arcsin(tirf_angle_sin)*180/np.pi
    self.tirf_angle_err = np.abs(tirf_angle_deriv) * self.d_exc_err
    
def fit_renorm_no_appr_fix_angle(
    self,
    p0: tuple[float] = (ALPHA_GUESS,),
):
    d_exc = self.params.lambda_exc/(4*np.pi)/np.sqrt(self.params.n_i**2*np.sin(np.radians(self.params.tirf_angle))**2 - self.params.n_s**2)
    self._calc_coll_fl_arr_axelrod()
    N_data = self.clust_means[:, :, 2]
    flat_z = self.z_real[:, 1:].ravel()
    # Normalize datal
    F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
    z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
    # Model function
    def F(z, alpha):
        num = (alpha * np.exp(-z / d_exc) + (1 - alpha))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
        den = (alpha * np.exp(-z_0 / d_exc) + (1 - alpha))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
        return num / den
    # Fit
    popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0],[ALPHA_MAX]))
    alpha_exc = popt[0]

    perr = np.sqrt(np.diag(pcov))
    self.alpha_exc = alpha_exc
    self.d_exc = d_exc
    self.alpha_exc_err = perr[0]
    self.d_exc_err = 0

    self.tirf_angle = self.params.tirf_angle
    self.tirf_angle_err = 0 
    
def fit_renorm_no_appr_fix_angle_biexp(
    self,
    p0: tuple[float] = (ALPHA_GUESS, D_LONG_GUESS),
):
    d_exc = self.params.lambda_exc/(4*np.pi)/np.sqrt(self.params.n_i**2*np.sin(np.radians(self.params.tirf_angle))**2 - self.params.n_s**2)
    self._calc_coll_fl_arr_axelrod()
    N_data = self.clust_means[:, :, 2]
    flat_z = self.z_real[:, 1:].ravel()
    # Normalize datal
    F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
    z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
    # Model function
    def F(z, alpha, d_long):
        num = (alpha * np.exp(-z / d_exc) + (1 - alpha) * np.exp(-z / d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
        den = (alpha * np.exp(-z_0 / d_exc) + (1 - alpha) * np.exp(-z / d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
        return num / den
    # Fit
    popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0, d_exc],[ALPHA_MAX, np.inf]))

    perr = np.sqrt(np.diag(pcov))
    self.alpha_exc = popt[0]
    self.d_exc = d_exc
    self.d_long = popt[1]
    self.d_long_err = perr[1]
    self.alpha_exc_err = perr[0]
    self.d_exc_err = 0

    print(self.d_long)

    self.tirf_angle = self.params.tirf_angle
    self.tirf_angle_err = 0 
    
def fit_no_appr_fix_angle_biexp_each_orig(
    self
):
    d_exc = self.params.lambda_exc/(4*np.pi)/np.sqrt(self.params.n_i**2*np.sin(np.radians(self.params.tirf_angle))**2 - self.params.n_s**2)
    self._calc_coll_fl_arr_axelrod()
    # Model function
    def N(z, alpha, N0, d_long):
        return N0*(alpha*np.exp(-z/d_exc) + (1 - alpha)*np.exp(-z/d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)/interp1d(Z_SIM_DISCR, self.coll_fl_discr)(0)
    # Fit
    self.alpha_arr = np.zeros(len(self.clust_means))
    self.N_0_arr = np.zeros(len(self.clust_means))
    self.d_long_arr = np.zeros(len(self.clust_means))
    for orig_idx in range(len(self.clust_means)):
        N_data = self.clust_means[orig_idx, :, 2].ravel()
        flat_z = self.z_real[orig_idx, :].ravel()
        p0 = (ALPHA_GUESS, flat_z[0], D_LONG_GUESS)
        popt, pcov = curve_fit(N, flat_z, N_data, p0=p0, bounds=([0, 0, d_exc],[ALPHA_MAX, np.inf, 2000]))
        self.alpha_arr[orig_idx] = popt[0]
        self.N_0_arr[orig_idx] = popt[1]
        self.d_long_arr[orig_idx] = popt[2]
        

    self.alpha_exc = np.mean(self.alpha_arr)
    self.alpha_exc_err = np.std(self.alpha_arr)
    self.d_exc = d_exc
    self.d_exc_err = 0

    self.tirf_angle = self.params.tirf_angle
    self.tirf_angle_err = 0
    
    self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
    self.N_0_avg = np.mean(self.N_0_arr)
    self.N_0_std = np.std(self.N_0_arr)
    
    self.d_long = np.mean(self.d_long_arr)
    self.d_long_err = np.std(self.d_long_arr)
    print(self.d_long)
    # variables for plot
    self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
    self.fit_func_forplot = N(self.z_ax_forplot, self.alpha_exc, 1, self.d_long)
    
def fit_renorm_no_appr_fix_alpha(
    self,
    p0: tuple[float] = (D_GUESS,),
):
    """Calculate SIMPLER effective params from known z and N.

    This function fits N instead of z and should be used when N0 of each
    origami is unknown.

    In this case, the fitting function is an exponential + constant multiplied by the CF,
    and alpha is fixed

    Parameters
    ----------
        p0: tuple[float, float], OPTIONAL
            Initial guesses for alpha_F and d_F

    TODO: add z_0 to account for constant linker added distance.
    """
    self._calc_coll_fl_arr_axelrod()
    N_data = self.clust_means[:, :, 2]
    flat_z = self.z_real[:, 1:].ravel()
    # Normalize datal
    F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
    z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
    # Model function
    def F(z, d_exc):
        num = (self.alpha_fixed * np.exp(-z / d_exc) + (1 - self.alpha_fixed))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
        den = (self.alpha_fixed * np.exp(-z_0 / d_exc) + (1 - self.alpha_fixed))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
        return num / den
    # Fit
    popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0],[np.inf]))
    d_exc = popt[0]

    perr = np.sqrt(np.diag(pcov))
    self.alpha_exc = self.alpha_fixed
    self.d_exc = d_exc
    self.alpha_exc_err = 0
    self.d_exc_err = perr[0]
    
    tirf_angle_lambda_factor = self.params.lambda_exc / (4*np.pi)
    tirf_angle_sqrt_factor = np.sqrt(((tirf_angle_lambda_factor / self.d_exc)**2 + self.params.n_s**2))
    tirf_angle_sin = tirf_angle_sqrt_factor / self.params.n_i
    tirf_angle_deriv = (1 / np.sqrt(1 - tirf_angle_sin**2)) * (1 / self.params.n_i) * (1/2) * (1 / tirf_angle_sqrt_factor) * tirf_angle_lambda_factor**2 * (2/self.d_exc**3) * (180/np.pi)
    self.tirf_angle = np.arcsin(tirf_angle_sin)*180/np.pi
    self.tirf_angle_err = np.abs(tirf_angle_deriv) * self.d_exc_err
    
def fit_N0_exp_appr(self):
    """
    This function should be called once alpha_F and d_F have already been fitted.
    It iterates over all selected origamis and fits N_0 (keeping alpha_F and d_F fixed!)
    for each one.
    """
    self.N_0_arr = np.zeros(len(self.clust_means), dtype=float)
    def F(z, N_0):
        return N_0*(self.alpha_F * np.exp(-z / self.d_F) + (1 - self.alpha_F))
    for orig_idx in range(len(self.clust_means)):
        z_data = self.z_real[orig_idx, :]
        N_data = self.clust_means[orig_idx, :, 2]
        N_0_val, N_0_err = curve_fit(F, z_data, N_data, p0=self.clust_means[orig_idx, 0, 2], bounds=([0],[np.inf]))
        self.N_0_arr[orig_idx] = N_0_val
    self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
    self.N_0_avg = np.mean(self.N_0_arr)
    self.N_0_std = np.std(self.N_0_arr)
    # variables for plot
    self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
    self.fit_func_forplot = F(self.z_ax_forplot, 1)
    
def fit_N0_no_appr(self):
    """
    It iterates over all selected origamis and fits N_0 (keeping other parameters fixed) for each one.
    """
    self.N_0_arr = np.zeros(len(self.clust_means), dtype=float)
    def F(z, N_0):
        return N_0*(self.alpha_exc * np.exp(-z / self.d_exc) + (1 - self.alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)/interp1d(Z_SIM_DISCR, self.coll_fl_discr)(0)
    for orig_idx in range(len(self.clust_means)):
        z_data = self.z_real[orig_idx, :]
        N_data = self.clust_means[orig_idx, :, 2]
        N_0_val, N_0_err = curve_fit(F, z_data, N_data, p0=self.clust_means[orig_idx, 0, 2], bounds=([0],[np.inf]))
        self.N_0_arr[orig_idx] = N_0_val
    self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
    self.N_0_avg = np.mean(self.N_0_arr)
    self.N_0_std = np.std(self.N_0_arr)
    # variables for plot
    self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
    self.fit_func_forplot = F(self.z_ax_forplot, 1)
    
def fit_N0_no_appr_biexp(self):
    """
    It iterates over all selected origamis and fits N_0 (keeping other parameters fixed) for each one.
    """
    self.N_0_arr = np.zeros(len(self.clust_means), dtype=float)
    def F(z, N_0):
        return N_0*(self.alpha_exc * np.exp(-z / self.d_exc) + (1 - self.alpha_exc) * np.exp(-z / self.d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)/interp1d(Z_SIM_DISCR, self.coll_fl_discr)(0)
    for orig_idx in range(len(self.clust_means)):
        z_data = self.z_real[orig_idx, :]
        N_data = self.clust_means[orig_idx, :, 2]
        N_0_val, N_0_err = curve_fit(F, z_data, N_data, p0=self.clust_means[orig_idx, 0, 2], bounds=([0],[np.inf]))
        self.N_0_arr[orig_idx] = N_0_val
    self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
    self.N_0_avg = np.mean(self.N_0_arr)
    self.N_0_std = np.std(self.N_0_arr)
    # variables for plot
    self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
    self.fit_func_forplot = F(self.z_ax_forplot, 1)
    
def backcalc_tirf_angle(self):
    """
    This function infers, from the global decay curve, the TIRF angle, using information about emission wavelength and objective NA
    """
    self.simpler_prof = self.alpha_F*np.exp(-Z_SIM_FIT_ARR/self.d_F) + (1 - self.alpha_F)
    self._calc_coll_fl_arr_axelrod()
    self.exc_prof = self.simpler_prof / self.params.coll_fl_interp_grid
    def F_exc(x, d_exc, b, c):
        return b * np.exp(-x/d_exc) + c                           
    popt, pcov = curve_fit(F_exc, Z_SIM_FIT_ARR, self.exc_prof, p0 = [200, 0.9, 0.1])
    self.d_exc = popt[0]
    self.d_exc_err = pcov[0, 0]
    self.exc_fit = F_exc(Z_SIM_FIT_ARR, popt[0], popt[1], popt[2])
    self.tirf_angle = np.arcsin(np.sqrt(((self.params.lambda_exc/(4*np.pi*popt[0]))**2 + self.params.n_s**2)/self.params.n_i**2))*180/np.pi
    
def backcalc_glob_param_biexp(self):
    """
    This function approximates the real decay with an exponential to get the global decay parameters
    """
    self.glob_prof = (self.alpha_exc*np.exp(-Z_SIM_FIT_ARR/self.d_exc) + (1 - self.alpha_exc)*np.exp(-Z_SIM_FIT_ARR/self.d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(Z_SIM_FIT_ARR)
    def F_F(z, d_F, alpha_F, norm):
        return norm*(alpha_F*np.exp(-z/d_F) + (1 - alpha_F))
    popt, pcov = curve_fit(F_F, Z_SIM_FIT_ARR, self.glob_prof, p0 = [200, 0.9, 0.1], bounds=([0, 0, 0],[np.inf, np.inf, np.inf]))
    self.d_F = popt[0]
    self.d_F_err = pcov[0, 0]
    self.alpha_F = popt[1]
    self.alpha_F_err = pcov[1, 1]
    
def z_from_N_appr(self, n_ph, n_ph_0):
    """
    This function computes z from N, given the global decay parameters
    """
    return self.d_F*np.log(self.alpha_F/((n_ph/n_ph_0) - (1 - self.alpha_F)))