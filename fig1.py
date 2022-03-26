# reference: W. H. Press
#   "Strong profiling is not mathematically optimal
#    for discovering rare malfeasors"
#   Proceedings of the National Academy of Sciences 106 (2009) 1716

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

def eff_infty(alpha):
    """ efficiency when N->inf """
    if alpha<2: return (2-alpha)/4
    elif alpha>2: return zeta(alpha-1)/zeta(alpha/2)**2
    else: return 0

alpha = np.linspace(0,4,41) # power law index
N = 1e4 # number of population

i = np.arange(1,N+1) # index of individuals
p = i[:,np.newaxis]**(-alpha) # prior probability
p /= np.sum(p, axis=0) # normalize

mu_A = np.dot(i,p) # mean number of trials in sampling without replacement
mu_D = np.sum(np.sqrt(p), axis=0)**2 # in weak sampling with replacement
E = [eff_infty(s) for s in alpha]

# plot efficiencies
plt.plot(alpha, mu_A/mu_D, 'b', label=r'weak, $N=10^4$')
plt.plot(alpha, mu_A/N, 'r', label=r'strong, $N=10^4$')
plt.plot(alpha, E, 'g--', label=r'weak, $N\to\infty$')

plt.legend(fontsize=14)
plt.axis([alpha[0], alpha[-1], 0, 0.5])
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel(r'$E$', fontsize=14)
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
