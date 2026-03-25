def px_to_nm(arr_toconv, px_size_nm):
    """
    This function converts x and y coordinates of a given array from px to nm
    """
    arr_toconv[:, 0] = arr_toconv[:, 0]*px_size_nm
    arr_toconv[:, 1] = arr_toconv[:, 1]*px_size_nm
    return arr_toconv