# -*- coding: utf-8 -*-
"""

@author: aszalai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import h5py
import h5py as h5
import yaml
import time
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from tqdm import tqdm


filename = "example_spectrin_large.hdf5"


#%% simpler parameters

# angle = 67.5
# alpha = 0.9
# lambdaex = 532
# lambdaem = 582
# N0 = 16000 #R1 13500, R2 16000
# ni = 1.51
# ns = 1.33
# NA = 1.42


px_size = 133 # nm
radius_threshold = 75 # nm

#%% load files

with pd.HDFStore(filename,'r') as store:
    data=store['locs']

#with h5py.File(filename, "r") as store:
#    data=store['locs']

#### read yaml file and store data in info
yaml_file = filename[:-5] + ".yaml"
with open(yaml_file, "r") as info_file:
    info = list(yaml.load_all(info_file, Loader=yaml.FullLoader)) 
#### save data of info



#%%  filter frames and perform SIMPLER

radius_threshold_px = radius_threshold/px_size
# start = time.time()

# frames = np.array(data['frame'])
# x = np.array(data['x'])
# y = np.array(data['y'])
# xy = np.column_stack((np.transpose(x),np.transpose(y)))
# discard_yn = np.ones((len(x),))


# for i in tqdm(range(len(x))):
#     if frames[i]>0 and frames[i]<np.max(frames):
#         idx_frame_minus_1 = np.where(frames == (frames[i]-1))
#         idx_frame_plus_1 = np.where(frames == (frames[i]+1))
        
#         if len(idx_frame_minus_1[0]) > 0 and len(idx_frame_plus_1[0]) > 0:
#             distance_frame_minus_1 = distance.cdist(np.reshape(xy[i,:],(1,2)), 
#                                                     xy[idx_frame_minus_1[0],:], 'euclidean')
#             distance_frame_minus_1 = np.sort(distance_frame_minus_1)
            
#             distance_frame_plus_1 = distance.cdist(np.reshape(xy[i,:],(1,2)), 
#                                                     xy[idx_frame_plus_1[0],:], 'euclidean')
#             distance_frame_plus_1 = np.sort(distance_frame_plus_1)
            
#             if distance_frame_minus_1[0,0] < radius_threshold_px and distance_frame_plus_1[0,0] < radius_threshold_px:
#                 discard_yn[i] = 0

                

# end1 = time.time()
# print('Time of filtering step (s):', end1-start)


start = time.time()

frames = np.array(data['frame'])
x = np.array(data['x'])
y = np.array(data['y'])
xy = np.column_stack((np.transpose(x), np.transpose(y)))
discard_yn = np.ones((len(x),))
framejump = np.nonzero(np.diff(frames, prepend=-np.inf, append=np.inf) != 0)[0]
for idxframe in tqdm(range(1, len(framejump)-2)):
    prevframe = frames[framejump[idxframe-1]]
    nextframe = frames[framejump[idxframe+1]]
    frame = frames[framejump[idxframe]]
    if (frame - 1 == prevframe) and (frame + 1 == nextframe):
        f_slice = slice(framejump[idxframe], framejump[idxframe+1])
        prev_slice = slice(framejump[idxframe-1], framejump[idxframe])
        next_slice = slice(framejump[idxframe+1], framejump[idxframe+2])
        distance_prev = distance.cdist(xy[f_slice], xy[prev_slice], 'euclidean')
        distance_next = distance.cdist(xy[f_slice], xy[next_slice], 'euclidean')
        has_prev = np.any(distance_prev < radius_threshold_px, axis=1)
        has_next = np.any(distance_next < radius_threshold_px, axis=1)
        discard_yn[f_slice][np.logical_and(has_prev, has_next)] = 0

end1 = time.time()
print('Time of filtering step nuevo (s):', end1-start)


# start = time.time()

# frames = np.array(data['frame'])
# x = np.array(data['x'])
# y = np.array(data['y'])
# xy = np.column_stack((np.transpose(x),np.transpose(y)))
# discard_ynbis = np.ones((len(x),))

# framejump = np.nonzero(np.diff(frames,prepend=-np.inf, append=np.inf)!=0)[0]
# for idxframe in tqdm(range(1, len(framejump)-2)):
#     prevframe = frames[framejump[idxframe-1]]
#     nextframe = frames[framejump[idxframe+1]]
#     frame = frames[framejump[idxframe]]
#     if frame - 1 == prevframe:
#         idx_frame_minus_1 = np.arange(framejump[idxframe-1], framejump[idxframe])
#     else:
#         idx_frame_minus_1 = np.empty((0,))
#     if frame + 1 == nextframe:
#         idx_frame_plus_1 = np.arange(framejump[idxframe+1], framejump[idxframe+2])
#     else:
#         idx_frame_plus_1 = np.empty((0,))
#     if len(idx_frame_minus_1) > 0 and len(idx_frame_plus_1) > 0:
#         for i in range(framejump[idxframe], framejump[idxframe+1]):
#             distance_frame_minus_1 = distance.cdist(np.reshape(xy[i,:],(1,2)), xy[idx_frame_minus_1,:], 'euclidean')
#             distance_frame_minus_1 = np.sort(distance_frame_minus_1)
#             distance_frame_plus_1 = distance.cdist(np.reshape(xy[i,:],(1,2)), xy[idx_frame_plus_1,:], 'euclidean')
#             distance_frame_plus_1 = np.sort(distance_frame_plus_1)
#             if distance_frame_minus_1[0,0] < radius_threshold_px and distance_frame_plus_1[0,0] < radius_threshold_px:
#                 discard_ynbis[i] = 0

# end1 = time.time()
# print('Time of filtering step bis (s):', end1-start)

# print(np.all(discard_ynbis == discard_yn))


idx_to_discard = np.where(discard_yn == 1)

data_filtered = data.drop(labels=idx_to_discard[0], axis=0)
data_filtered = data_filtered.reset_index(level=None, drop=True, inplace=False,
                                          col_level=0)


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


sa, saType = df_to_sarray(data_filtered)
# Open/create the HDF5 file
f = h5py.File(filename[:-5]+'_frames_filtered.hdf5', 'a')
# Save the structured array
f.create_dataset('locs', data=sa, dtype=saType)
f.close()

# info[0]['Frames'] = np.max(data_filtered['frame'])

with open(filename[:-5]+'_frames_filtered.yaml', "w") as file:
    yaml.dump_all(info, file, default_flow_style=False)

end2 = time.time()
print('Script total time (s):', end2-start)
