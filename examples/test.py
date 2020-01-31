import numba as nb
import numpy as np

@nb.njit(parallel=True)
def numbafy(a, b):
	return np.array((a, b))

a = np.array([1, 2])
b = np.array([3, 4])
c = numbafy(a, b)
print(c)