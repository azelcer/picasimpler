import numpy as np

means = np.array((
    (4, 5, 15000),
    (3, 5, 30000),
    (6, 5, 5000),
    (4, 6, 25000)
))

sigmas = np.array((
    (3, 2, 150),
    (2, 1, 300),
    (4, 5, 50),
    (2, 2, 250)
))

def reord_idx(means):
    return np.argsort(-means[:, 2])

def reord(means, sigmas):
    return list(zip(*sorted(zip(means, sigmas), key=lambda pair: pair[0][2])))
        
permut_idx = reord_idx(means)

print(permut_idx)