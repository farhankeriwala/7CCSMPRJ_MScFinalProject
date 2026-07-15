import numpy as np

from env.hedging_env import HedgingEnv
from env.merton import MertonSimulator

class BlackScholesBaselineAgent:
    def __init__(self, env: HedgingEnv):
        self.env = env

    def select_action( self, observations: np.ndarray ) -> np.ndarray:
        """
        Selects the hedge ratio based on the current state
        :param observations:
        :return: the hedge ratio
        """

        S_r = observations[ 0 ]
        time_fraction = observations[ 1 ]

        S = S_r * self.env.S0
        K = self.env.K
        tau = time_fraction * self.env.T

        if tau < 1e-8:
            delta_hedge = 1.0 if S > K else 0.0
        else:
            delta_hedge = self.env.simulator.black_scholes_delta( S, K ,tau )

        return np.array( [ delta_hedge, 0.0 ], dtype=np.float32 )

    def run_ep(self) -> dict:
        observations, _ = self.env.reset()
        total_reward = 0.0
        intermediate_rewards = []

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = self.select_action(observations)
            observations, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            intermediate_rewards.append(reward)

        return {
            "total_pnl": total_reward,
            "intermediate_rewards": intermediate_rewards,
            "knocked_out": info["knocked_out"],
        }
