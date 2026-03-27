from enum import auto

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
    if type(prog_order) is auto:
        return len(enum_cls.__members__) + 1
    else:
        raise TypeError(f"First element of Enum member must be auto(), got {type(prog_order).__name__!r}")
    
def safe_float(expr: str):
    """
    This function converts strings to floats as would float do, but add the empty string to 0 case
    """
    if (expr is None) or (expr==""):
        return 0
    else:
        return float(expr)