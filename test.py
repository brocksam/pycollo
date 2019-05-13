import numba as nb
import numpy as np

@nb.jit
def reshape_x(x):
	a = x[:, 0:4].reshape(2, -1)
	b = x[:, 4:5]
	lst = []
	for row in a:
		lst.append(row)
	for row in b:
		lst.append(row)
	return lst

x_data = np.array([range(5)])
print(x_data)
# result_list_comp = reshape_x_list_comp(x_data)
# result_star_unpack = reshape_x_star_unpack(x_data)
result = reshape_x(x_data)
# print(result_list_comp)
# print(result_star_unpack)
print(result)