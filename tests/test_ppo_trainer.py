import torch
import numpy as np
import pytest
from env.vec_hedging_env import VecHedgingEnv
from agent.actor_critic import ActorCritic
from training.ppo_trainer import PPOTrainer


@pytest.fixture
def trainer():
    """
    Small trainer for fast smoke testing — reduced dimensions throughout.
    Not a full training run, just confirms the plumbing works end-to-end.
    """
    env = VecHedgingEnv(N=16)   # small N for speed

    net = ActorCritic(observation_dim=4, action_dim=2, hidden_dim=64)

    trainer = PPOTrainer(
        env          = env,
        net          = net,
        lr           = 3e-4,
        gamma        = 1.0,
        gae_lambda   = 0.95,
        epsilon_clip = 0.2,
        c_val      = 0.5,
        c_entropy    = 0.001,
        num_epochs     = 2,         # reduced for speed
        batch_size   = 32,        # reduced for speed
        rollout_steps = 100,      # reduced for speed
    )
    return trainer


# ------------------------------------------------------------------
# Test 1 — Rollout collection runs without error and returns correct types
# ------------------------------------------------------------------

def test_rollout_collection(trainer):
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, episode_returns = trainer._collect_rollout(obs)

    # obs should be a tensor of correct shape
    assert obs.shape == (trainer.env.N, trainer.env.observations_dim), \
        f"obs shape wrong: {obs.shape}"

    # last_value should be (N,)
    assert last_value.shape == (trainer.env.N,), \
        f"last_value shape wrong: {last_value.shape}"

    # episode_returns should be a non-empty list of floats
    assert isinstance(episode_returns, list), "episode_returns should be a list"
    assert len(episode_returns) > 0, "No episodes completed during rollout"
    assert all(isinstance(r, float) for r in episode_returns), \
        "episode_returns should contain floats"


# ------------------------------------------------------------------
# Test 2 — GAE returns correct shapes and finite values
# ------------------------------------------------------------------

def test_gae_computation(trainer):
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)

    advantages, returns = trainer._compute_gae(
        trainer.buffer_rewards,
        trainer.buffer_vals,
        trainer.buffer_dones,
        last_value,
    )

    # Shapes must match rollout buffer dimensions
    assert advantages.shape == (trainer.rollout_steps, trainer.env.N), \
        f"advantages shape wrong: {advantages.shape}"
    assert returns.shape == (trainer.rollout_steps, trainer.env.N), \
        f"returns shape wrong: {returns.shape}"

    # No nan or inf — would silently corrupt PPO updates
    assert torch.all(torch.isfinite(advantages)), "Non-finite advantages detected"
    assert torch.all(torch.isfinite(returns)),    "Non-finite returns detected"


# ------------------------------------------------------------------
# Test 3 — PPO update runs without error and returns loss dict
# ------------------------------------------------------------------

def test_ppo_update(trainer):
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)

    advantages, returns = trainer._compute_gae(
        trainer.buffer_rewards,
        trainer.buffer_vals,
        trainer.buffer_dones,
        last_value,
    )

    losses = trainer._ppo_update(advantages, returns)

    # Check loss dict has correct keys
    assert "policy_loss"  in losses
    assert "value_loss"   in losses
    assert "entropy"      in losses

    # All losses should be finite scalars
    assert np.isfinite(losses["policy_loss"]),  "policy_loss is not finite"
    assert np.isfinite(losses["value_loss"]),   "value_loss is not finite"
    assert np.isfinite(losses["entropy"]),      "entropy is not finite"


# ------------------------------------------------------------------
# Test 4 — Network weights change after one PPO update
# ------------------------------------------------------------------

def test_weights_update(trainer):
    """
    Confirms gradient actually flows and updates parameters.
    If weights don't change, the optimiser or loss is broken.
    """
    # Snapshot weights before update
    weights_before = {
        name: param.clone()
        for name, param in trainer.net.named_parameters()
    }

    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)
    advantages, returns = trainer._compute_gae(
        trainer.buffer_rewards,
        trainer.buffer_vals,
        trainer.buffer_dones,
        last_value,
    )
    trainer._ppo_update(advantages, returns)

    # At least some parameters must have changed
    any_changed = False
    for name, param in trainer.net.named_parameters():
        if not torch.allclose(weights_before[name], param):
            any_changed = True
            break

    assert any_changed, "No network weights changed after PPO update"


# ------------------------------------------------------------------
# Test 5 — Full train() loop runs for a small number of episodes
# ------------------------------------------------------------------

def test_train_loop(trainer):
    """
    Smoke test: run train() for 500 episodes and confirm it returns
    a non-empty list of episode returns without crashing.
    """
    returns = trainer.train(total_ep=500)

    assert isinstance(returns, list), "train() should return a list"
    assert len(returns) > 0,          "train() returned empty returns list"
    assert all(np.isfinite(r) for r in returns), \
        "Non-finite returns detected in training output"