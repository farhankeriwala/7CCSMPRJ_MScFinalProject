from env.merton import MertonSimulator
import numpy as np


class VecHedgingEnv:
    """
    A vectorised version of the HedgingEnv for multiple paths ensuring more efficient computation.
    """

    def __init__(self, N: int = 256, S0: float = 100.0, K: float = 100.0, B: float = 75.0, r: float = 0.05,
                 sigma: float = 0.20, lam: float = 1.5, mu_J=-0.04, sigma_J: float = 0.08, T: float = 1.0,
                 num_steps: int = 50, transaction_cost: float = 0.001, rho: float = 0.5, lam_override: float = None):
        self.N = N
        self.S0 = S0
        self.K = K
        self.B = B
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.T = T
        self.num_steps = num_steps
        self.transaction_cost = transaction_cost
        self.rho = rho
        self.lam_override = lam_override

        # compute the time step and set the observation and action dimensions
        self.dt = T / num_steps
        self.observations_dim = 4
        self.actions_dim = 2

        # initialize the price path simulator
        self.simulator = MertonSimulator(S0, r, sigma, lam, mu_J, sigma_J, T, num_steps)

        # allocate the prices and sentiments
        self.prices = np.zeros((N, num_steps + 1), dtype=np.float64)
        self.sentiment = np.zeros((N, num_steps + 1), dtype=np.float64)

        # env state
        self.t = 0
        self.prev_delta = np.zeros((N, 2), dtype=np.float32)
        self.knocked_out = np.zeros(N, dtype=np.bool)

    def _simulate_price_paths(self):
        """
        Batch path simulator
        :return:

        """

        # overide lam for curriculum learning
        lam = self.lam if self.lam_override is None else self.lam_override

        # compute the adjusted diffusion coefficient
        k = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1.0

        dt = self.dt

        # set the initial prices
        self.prices = np.zeros((self.N, self.num_steps+1), dtype=np.float64)

        self.prices[:, 0] = self.S0

        for step in range(self.num_steps):
            # set the current prices for all paths
            S_t = self.prices[:, step]

            # generate the diffusion process
            epsilon = np.random.standard_normal(self.N)
            diff = (self.r - 0.5 * self.sigma ** 2 - lam * k) * dt + self.sigma * np.sqrt(dt) * epsilon

            # create the jump process and the contribution of the jump
            jump_occur = np.random.random(self.N) < lam * dt
            J = np.random.normal(self.mu_J, self.sigma_J, self.N)
            jump_contrib = np.where(jump_occur, J, 0.0)

            # set the next prices for all paths using the diffusion and jump processes
            exponent = np.clip(diff + jump_contrib, -500.0, 500.0)
            self.prices[:, step + 1] = np.maximum(S_t * np.exp(exponent), 1e-8)

    def _generate_sentiment_signals(self):
        """
        generate all the sentiment signals for all paths
        :return:
        """

        # compute log returns
        log_ret = np.zeros_like(self.prices)
        log_ret[:, 1:] = np.log(self.prices[:, 1:] / self.prices[:, :-1])

        # compute the threshold for the signals
        threshold = 2.0 * self.sigma * np.sqrt(self.dt)

        jump_flags = (np.abs(log_ret) > threshold).astype(int)
        informative = jump_flags * 2.0 - 1.0
        noise = np.random.uniform(-1.0, 1.0, size=self.prices.shape).astype(np.float32)

        self.sentiment = np.clip(self.rho * informative + (1.0 - self.rho) * noise, -1.0, 1.0)

    def reset(self):
        """
        function to reset all N environments simultaneously
        :return: observatins of shape N,4
        """
        self._simulate_price_paths()
        self._generate_sentiment_signals()
        self.t = 0
        self.prev_delta = np.zeros((self.N, 2), dtype=np.float32)
        self.knocked_out = np.zeros(self.N, dtype=np.bool)

        return self._get_observations()

    def step(self, actions: np.ndarray):
        actions = np.clip(actions, 0.0, 1.0).astype(np.float32)
        S_t = self.prices[:, self.t]
        S_tp1 = self.prices[:, self.t + 1]
        dS = S_tp1 - S_t

        # combined equity and hedge positions
        pnl = (actions[:, 0] + actions[:, 1]) * dS

        # compute the transaction cost and adjust the reward
        delta_change = np.abs(actions - self.prev_delta)
        transaction_cost = self.transaction_cost * delta_change.sum(axis=1) * S_t

        rewards = (pnl - transaction_cost).astype(np.float32)

        self.prev_delta = actions.copy()
        self.t += 1

        new_knocked_out = (~self.knocked_out) & (S_tp1 <= self.B)
        self.knocked_out |= new_knocked_out

        terminated = (self.t == self.num_steps)
        if terminated:
            adjusted_terminal = self._terminal_adjustment(S_tp1)
            rewards += adjusted_terminal
            dones = np.ones(self.N, dtype=np.bool)
        else:
            dones = np.zeros(self.N, dtype=np.bool)

        obsevations = self._get_observations()

        info = {
            "pnl": pnl,
            "transaction_cost": transaction_cost,
            "knocked_out": self.knocked_out,
        }

        return obsevations, rewards, dones, info

    def _get_observations(self) -> np.ndarray:
        S_t = self.prices[:, self.t]
        time_remaining = (self.T - self.t * self.dt) / self.T
        z_t = self.sentiment[:, self.t]

        # stack of observations such that each row is one observation
        observations = np.stack([
            S_t / self.S0,
            np.full(self.N, time_remaining, dtype=np.float32),
            self.prev_delta[:, 0],
            z_t
        ], axis=1)

        return observations.astype(np.float32)

    def _terminal_adjustment(self, S_T: np.ndarray) -> np.ndarray:
        """
        This function adjusts the rewards for the terminal states for the option payoff for all N paths
        :param S_T: all N terminal stock prices
        :return: adjusted rewards
        """
        payoff = np.maximum(S_T - self.K, 0.0)
        adjusted_payoff = np.where(self.knocked_out, 0.0, -payoff)
        return adjusted_payoff.astype(np.float32)

    def set_lam_override(self, lam: float):
        """
        function to override the diffusion coefficient for the curriculum learning
        :param lam:
        :return:
        """
        self.lam_override = lam
