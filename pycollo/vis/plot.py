import matplotlib.pyplot as plt

def plot_simulation(self, *, q_flag=False, u_flag=False):
		if q_flag:
			for q in self.q_ind_list:
				plt.plot(self.times, self.states[q])
		elif u_flag:
			for u in self.u_ind_list:
				plt.plot(self.times, self.states[u])
		plt.show()
	
