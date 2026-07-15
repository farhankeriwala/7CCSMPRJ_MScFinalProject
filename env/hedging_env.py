import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.merton import MertonSimulator


class HedgingEnv(gym.Env):
    """
    """

    metadata = {"render_modes": []}

    def __init__(self, S0: float = 100.0, K: float = 100.0, B: float = 75.0, r: float = 0.05, sigma: float = 0.20,
                 lam: float = 1.5, mu_J: float = -0.04, sigma_J: float = 0.08, T: float = 1.0, num_steps: int = 50,
                 transaction_cost: float = 0.001, rho: float = 0.5, lam_override: float = None):
        super().__init__()

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

        # Compute the time step
        self.dt = T / num_steps

        # define the price path simulator
        self.simulator = MertonSimulator(S0, r, sigma, lam, mu_J, sigma_J, T, num_steps)

        # define the observation space (S/S0, (T-t)/T, prev_delta, z)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -3.0], dtype=np.float32),
            high=np.array([5.0, 1.0, 1.0, 3.0], dtype=np.float32),
            dtype=np.float32
        )

        # define the action space [delta_hedge, delta_equity]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.prices = None
        self.t = None
        self.prev_delta = None
        self.knocked_out = None
        self.sentiment = None

    def generate_sentiment_signal(self, prices: np.ndarray) -> np.ndarray:
        """
        This function generates a signal based on the jump arrivals which is detected via a large log return threshold.
        :param prices: the price path
        :return: a clipped sentiment signal
        """

        # compute the log returns
        n = len(prices)
        log_returns = np.zeros(n)
        log_returns[1:] = np.log(prices[1:] / prices[:-1])

        # compute the jump arrivals
        jump_threshold = 2.0 * self.sigma * np.sqrt(self.dt)
        jump_flags = (np.abs(log_returns) > jump_threshold).astype(int)

        # sample a noise signal
        noise = np.random.uniform(-1.0, 1.0, n)

        # informative signal is 1 if the jump occurred and -1 otherwise
        informative = jump_flags * 2.0 - 1.0

        # combine the informative and noise signals
        signal = self.rho * informative + (1.0 - self.rho) * noise

        # return a clipped signal
        return np.clip(signal, -1.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        lam = self.lam if self.lam_override is None else self.lam_override

        # generate the price path
        self.prices, _ = self.simulator.simulate_price_path(lam_override=lam)
        # generate the sentiment signals
        self.sentiment = self.generate_sentiment_signal(self.prices)

        self.t = 0
        self.prev_delta = np.array([0.0, 0.0], dtype=np.float32)
        self.knocked_out = False

        observations = self._get_observations()
        info = {}
        return observations, info

    def step(self, action: np.ndarray):
        assert not self.knocked_out or self.t < self.num_steps, "step() called after the terminal episode state was reached."

        # sample an action from the action space and get the delta for the hedge and equity positions
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        delta_hedge = action[0]
        delta_equity = action[1]

        # get the prices and compute the price change for the pnl
        S_t = self.prices[self.t]
        S_tp1 = self.prices[self.t + 1]
        dS = S_tp1 - S_t

        pnl = delta_hedge * dS - delta_equity * dS

        transaction_cost = self.transaction_cost * np.sum(np.abs(action - self.prev_delta)) * S_t
        reward = pnl - transaction_cost

        # update interanl state
        self.t += 1
        self.prev_delta = action

        # if the barrier has been crossed set the knocked out flag to True
        if S_tp1 <= self.B:
            self.knocked_out = True

        terminated = (self.t == self.num_steps)
        truncated = False

        if terminated:
            reward += self._terminal_adjustment(S_tp1)

        observations = self._get_observations()
        info = {
            "S_tp1": S_tp1,
            "knocked_out": self.knocked_out,
            "pnl": pnl,
            "transaction_cost": transaction_cost,
        }

        return observations, reward, terminated, truncated, info

    def _get_observations(self) -> np.ndarray:

        # get the current prices, time remaining and sentiment signal
        S_t = self.prices[self.t]
        time_remaining = (self.T - self.t * self.dt) / self.T
        z_t = self.sentiment[self.t]

        # get the current observations
        observations = np.array([
            S_t / self.S0, time_remaining, self.prev_delta[0], z_t
        ], dtype=np.float32)

        return observations

    def _terminal_adjustment(self, S_T: float) -> float:
        """
        function to adjust the reward for the terminal state for the option payoff
        :param S_T: terminal stock price
        :return: adjusted reward
        """

        # check if option was already knocked out
        if self.knocked_out:
            return 0.0

        else:
            # negate as we are short the barrier option
            return -max(S_T - self.K, 0.0)
