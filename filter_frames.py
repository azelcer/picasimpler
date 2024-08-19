# -*- coding: utf-8 -*-
"""

@author: aszalai, azelcer
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance
import h5py
import yaml
import time
from tqdm import tqdm
import logging as _lgn

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)

filename = "/home/azelcer/Dropbox/2024/simpler/example_spectrin_large.hdf5"


# angle = 67.5
# alpha = 0.9
# lambdaex = 532
# lambdaem = 582
# N0 = 16000 #R1 13500, R2 16000
# ni = 1.51
# ns = 1.33
# NA = 1.42


def filter_data(data, radius_threshold, px_size) -> np.ndarray:
    """Filter localizations according to SIMPLER criteria."""
    r_th_sq = (radius_threshold/px_size)**2
    start = time.time()
    frames = np.array(data['frame'])
    xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
    discard_yn = np.ones((len(xy),), dtype=np.uint64)
    framejump = np.nonzero(np.diff(frames, prepend=-np.inf, append=np.inf) != 0)[0]
    distance_next = None
    for idxframe in tqdm(range(1, len(framejump)-2)):
        prevframe = frames[framejump[idxframe-1]]
        nextframe = frames[framejump[idxframe+1]]
        frame = frames[framejump[idxframe]]
        if (frame + 1 != nextframe):
            distance_next = None
            continue
        if (frame - 1 != prevframe):
            continue
        f_slice = slice(framejump[idxframe], framejump[idxframe+1])
        prev_slice = slice(framejump[idxframe-1], framejump[idxframe])
        next_slice = slice(framejump[idxframe+1], framejump[idxframe+2])
        if distance_next is not None:
            distance_prev = distance_next.T
        else:
            distance_prev = distance.cdist(xy[f_slice], xy[prev_slice], 'sqeuclidean')
        distance_next = distance.cdist(xy[f_slice], xy[next_slice], 'sqeuclidean')
        has_prev = np.any(distance_prev < r_th_sq, axis=1)
        has_next = np.any(distance_next < r_th_sq, axis=1)
        discard_yn[f_slice][np.logical_and(has_prev, has_next)] = 0
    end = time.time()
    print('Time of filtering step nuevo (s):', end-start)
    idx_to_discard = np.where(discard_yn == 1)
    return idx_to_discard


def calculate_z(data, alpha, df, N0) -> np.ndarray:
    """Calculate z according to SIMPLER criteria.

    We use fixed N0 for now.
    """
    # N0 = _calculate_N0(data, params)
    max_photons = np.max(data['photons'])
    min_alpha = 1 - (np.min(data['photons']) / max_photons)
    if alpha < min_alpha:
        _lgr.warning("Some location intensities are below non-evanescent excitation expected intensity")
    if N0 < max_photons:
        _lgr.warning("Some location intensities are above z=0 intensity")
    rv = df * np.log(alpha/(data['photons']/N0 - (1 - alpha)))
    return rv


def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int) 
                    col_type = ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise
    v = df.values
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype


if __name__ == '__main__':
    start = time.time()
    with pd.HDFStore(filename, 'r') as store:
        data = store['locs']

    yaml_file = filename[:-5] + ".yaml"
    with open(yaml_file, "r") as info_file:
        info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
    # px_size = 133 # nm
    px_size = info[1]['Pixelsize']
    radius_threshold = 75  # nm
    idx_to_discard = filter_data(data, radius_threshold, px_size)
    data_filtered = data.drop(labels=idx_to_discard[0], axis=0)
    data_filtered = data_filtered.reset_index(level=None, drop=True, inplace=False,
                                              col_level=0)
    print("minimum alpha is:", 1 - (np.min(data['photons']) / np.max(data['photons'])))
    z = calculate_z(data_filtered, 0.95, 100, np.max(data['photons']))
    data_filtered['z'] = z
    # save data of info
    sa, saType = df_to_sarray(data_filtered)
    # Open/create the HDF5 file
    f = h5py.File(filename[:-5]+'_frames_filtered.hdf5', 'a')
    # Save the structured array
    f.create_dataset('locs', data=sa, dtype=saType)
    f.close()
    with open(filename[:-5]+'_frames_filtered.yaml', "w") as file:
        yaml.dump_all(info, file, default_flow_style=False)
    end = time.time()
    print('Script total time (s):', end - start)
