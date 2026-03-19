means = (
    (4, 5, 15000),
    (3, 5, 30000),
    (6, 5, 5000),
    (4, 6, 25000)
)

sigmas = (
    (3, 2, 150),
    (2, 1, 300),
    (4, 5, 50),
    (2, 2, 250)
)

def reord(means, sigmas):
    return list(zip(*sorted(zip(means, sigmas), key=lambda pair: pair[0][2])))
        
means_ord, sigmas_ord = reord(means, sigmas)

print(means_ord)
print(sigmas_ord)