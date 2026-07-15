import numpy as np
from enum import auto

def gauss(xx, ampl, mu, sigma):
    return ampl*np.exp(-(xx - mu)**2/(2*sigma**2))/np.sqrt(2*np.pi)/sigma

def cleanup_z_score(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    # modified z-score, robust to outliers (using MAD instead of std)
    mod_z = 0.6745 * (data - median) / mad
    clean_data = data[np.abs(mod_z) < 3.5]  # common threshold
    return clean_data

def calc_tirf_angle_werr(d_exc, d_exc_err, lambda_exc, n_i, n_s):
    tirf_angle_lambda_factor = lambda_exc / (4*np.pi)
    tirf_angle_sqrt_factor = np.sqrt(((tirf_angle_lambda_factor / d_exc)**2 + n_s**2))
    tirf_angle_sin = tirf_angle_sqrt_factor / n_i
    tirf_angle_deriv = (1 / np.sqrt(1 - tirf_angle_sin**2)) * (1 / n_i) * (1/2) * (1 / tirf_angle_sqrt_factor) * tirf_angle_lambda_factor**2 * (2/d_exc**3) * (180/np.pi)
    tirf_angle = np.arcsin(tirf_angle_sin)*180/np.pi
    tirf_angle_err = np.abs(tirf_angle_deriv) * d_exc_err
    return tirf_angle, tirf_angle_err

def px_to_nm(arr_toconv, px_size_nm):
    """
    This function converts x and y coordinates of a given array from px to nm
    """
    arr_toconv[:, 0] = arr_toconv[:, 0]*px_size_nm
    arr_toconv[:, 1] = arr_toconv[:, 1]*px_size_nm
    return arr_toconv

def hex_to_rgba(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h)==6:
        return tuple([int(h[i:i+2], 16) for i in (0, 2, 4)]+[255])
    elif len(h)==8:
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4, 6))
    else:
        raise ValueError("input not in standard hex format for color")
    
def cust_auto(enum_cls, prog_order):
    """
    Custom auto function for automatic ordering of Enum class members, in case they are defined as tuples
    """
    if type(prog_order) in [auto, int]:
        return len(enum_cls.__members__) + 1
    else:
        raise TypeError(f"First element of Enum member must be auto() or int, got {type(prog_order).__name__!r}")
    
def safe_float_to0(expr: str):
    """
    This function converts strings to floats as would float do, but add the empty string to 0 case
    """
    if (expr is None) or (expr==""):
        return 0
    else:
        return float(expr)
    
def safe_float_tonone(expr: str):
    """
    This function converts strings to floats as would float do, but add the empty string to None case
    """
    if (expr is None) or (expr==""):
        return None
    else:
        return float(expr)
