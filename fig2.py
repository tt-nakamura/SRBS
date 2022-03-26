# reference: W. H. Press
#   "Strong profiling is not mathematically optimal
#    for discovering rare malfeasors"
#   Proceedings of the National Academy of Sciences 106 (2009) 1716

import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(0,4,41) # power law exponent
N = 1e4 # number of population
s = [0.25, 0.5, 0.75] # probability of recognizing malfeasors
M = [30, 25, 20] # number of trials

i = np.arange(1,N+1) # index of individuals
p = i[:,np.newaxis]**(-alpha) # prior probability
p /= np.sum(p, axis=0) # normalize
# mean number of trials in weak sampling with replacement
mu_D = np.sum(np.sqrt(p), axis=0)**2

p = np.expand_dims(p.T,-1)

for s,M in zip(s,M):
    q = p*(1-s)**np.arange(M)
    q = q.reshape(len(alpha), -1)
    q = np.sort(q.T, axis=0)[::-1]
    i = np.arange(1, len(q)+1)
    # mean number of trials in sampling without replacement
    mu_A = np.dot(i,q)*s**2
    # plot efficiency
    plt.plot(alpha, mu_A/mu_D, label='$s=%g$'%s)

plt.axis([alpha[0], alpha[-1], 0, 1])
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$E$', fontsize=14)
plt.text(3.3, 0.65, r'$N=10^4$', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
