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

    def simulate_price_path(self, lam_override=None):
        """
        This function will simulate a single path of the price process using the Merton model with the given parameters.
        :param lam_override: overrides the default lambda value
        :return: prices, jump_occurred
        """
        # overrides the default lambda value
        lam = self.lam if lam_override is None else lam_override

        # draw diffusion shocks from a normal distribution
        Z = np.random.normal( 0, 1, self.num_steps )

        # draw jump shocks
        jump_occurred = np.random.binomial( 1,min(lam * self.dt, 1.0), self.num_steps ).astype(bool)

        # draw jump sizes
        jump_sizes = np.random.normal(self.mu_J, self.sigma_J, self.num_steps)

        # log returns

        drift_diffusion = (
            ( self.r - 0.5 * self.sigma**2 - lam * self.k )* self.dt + self.sigma * np.sqrt( self.dt ) * Z
        )

        # the component is 0 if the jump did not occur and J otherwise
        jump_component = np.where( jump_occurred, jump_sizes, 0 )

        # compute log return as the sum of the drift and jump components
        log_returns = drift_diffusion + jump_component

        # price path
        prices = np.empty(self.num_steps + 1)
        prices[0] = self.S0
        prices[1:] = self.S0 * np.exp(np.cumsum(log_returns))

        return prices, jump_occurred

    def simulate_price_paths( self, num_paths, lam_override=None ):
        """
        This function will simulate many paths of the price process simultaneously using a vectorised implementation.
        :param num_paths: the number of paths to simulate
        :param lam_override: overrides the default lambda value
        :return: prices, jump_occurred
        """

        lam = self.lam if lam_override is None else lam_override

        Z = np.random.normal( 0, 1, ( num_paths, self.num_steps ) )

        jump_occurred = np.random.binomial(1, min(lam * self.dt, 1.0), ( num_paths, self.num_steps )).astype(bool)

        jump_sizes = np.random.normal( self.mu_J, self.sigma_J, (num_paths, self.num_steps ))

        drift_diffusion = (( self.r - 0.5 * self.sigma**2 - lam * self.k )*self.dt + self.sigma * np.sqrt( self.dt ) * Z)

        jump_component = np.where(jump_occurred, jump_sizes, 0)

        log_returns = drift_diffusion + jump_component

        prices = np.empty(( num_paths, self.num_steps + 1 ))
        prices[:,0] = self.S0
        prices[:,1:] = self.S0 * np.exp(np.cumsum(log_returns, axis=1))

        return prices, jump_occurred
