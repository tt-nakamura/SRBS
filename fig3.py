# reference: W. H. Press
#   "Strong profiling is not mathematically optimal
#    for discovering rare malfeasors"
#   Proceedings of the National Academy of Sciences 106 (2009) 1716

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import lognormal

alpha = 1 # power law index
N = 1e4 # number of population
sigma = np.linspace(0,5,65) # standard deviation of lognormal distribution

i = np.arange(1,N+1) # index of individuals
q = i[:,np.newaxis]**(-alpha) # estimated prior probability
q /= np.sum(q, axis=0) # normalize
p = q*lognormal(0, sigma, (len(q),len(sigma))) # prior probability
p /= np.sum(p, axis=0) # normalize
r = np.sqrt(q) # weak sampling probability
r /= np.sum(r, axis=0) # normalize

mu_A = np.dot(i,p) # mean number of trials in samling without replacement
mu_1 = np.sum(p/q, axis=0) # in strong sampling with replacement
mu_2 = np.sum(p/r, axis=0) # in weak sampling with replacement

# plot efficiencies
plt.plot(sigma, mu_A/mu_2, 'b', label=r'weak')
plt.plot(sigma, mu_A/mu_1, 'r', label=r'strong')
plt.plot(sigma, mu_A/N, 'g', label=r'random')

plt.legend(fontsize=14, loc='upper left')
plt.axis([sigma[0], sigma[-1], 0, 0.5])
plt.xlabel(r'$\sigma$', fontsize=14)
plt.ylabel(r'$E$', fontsize=14)
plt.text(0.1, 0.33, r'$\alpha=1$, $N=10^4$', fontsize=14)
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
