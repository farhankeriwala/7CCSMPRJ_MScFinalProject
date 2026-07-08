"""
This file implements the Merton jump diffusion dynamics for the price simulator. It simulates the stock price and
provides an analytical pricing equation for validation
"""

import numpy as np
from scipy.stats import norm

from math import factorial

class MertonSimulator:
    def __init__( self, S0, r, sigma, lam, mu_J, sigma_J, T, num_steps ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.T = T
        self.num_steps = num_steps
        self.dt = T / num_steps

        # drift correction
        self.k = np.exp( mu_J + 0.5 * sigma_J**2 ) - 1
