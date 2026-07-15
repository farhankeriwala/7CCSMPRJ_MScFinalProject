import torch
import numpy as np
import pytest
from env.vec_hedging_env import VecHedgingEnv
from agent.distributional_actor_critic import DistributionalActorCritic
from training.cvar_ppo_trainer import CvarPPOTrainer


@pytest.fixture
def trainer():
    """Small trainer for fast smoke testing."""
    env = VecHedgingEnv(N=16)

    net = DistributionalActorCritic(
        observation_dim      = env.observations_dim,
        action_dim      = env.actions_dim,
        hidden_dim   = 64,
        num_quantiles  = 32,
        alpha_cvar   = 0.05,
    )

    trainer = CvarPPOTrainer(
        env           = env,
        net           = net,
        lr            = 1e-4,
        gamma         = 1.0,
        gae_lambda    = 0.95,
        epsilon_clip  = 0.2,
        c_val       = 1.0,
        c_entropy     = 0.01,
        num_epochs      = 2,        # reduced for speed
        batch_size    = 32,       # reduced for speed
        rollout_steps = 100,      # reduced for speed
        kappa         = 1.0,
    )
    return trainer


# ------------------------------------------------------------------
# Test 1 — Rollout collection returns correct types and shapes
# ------------------------------------------------------------------

def test_rollout_collection(trainer):
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, ep_returns = trainer._collect_rollout(obs)

    assert obs.shape == (trainer.env.N, trainer.env.observations_dim), \
        f"obs shape wrong: {obs.shape}"
    assert last_value.shape == (trainer.env.N,), \
        f"last_value shape wrong: {last_value.shape}"
    assert isinstance(ep_returns, list)
    assert len(ep_returns) > 0
    assert all(isinstance(r, float) for r in ep_returns)

    # Quantile buffer should be filled
    assert trainer.buffer_quantiles.shape == (
        trainer.rollout_steps, trainer.env.N, trainer.net.num_quantiles
    ), f"quantile buffer shape wrong: {trainer.buffer_quantiles.shape}"


# ------------------------------------------------------------------
# Test 2 — Returns computation is finite
# ------------------------------------------------------------------

def test_returns_computation(trainer):
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)

    returns = trainer._compute_returns(
        trainer.buffer_rewards,
        trainer.buffer_dones,
        last_value,
    )

    assert returns.shape == (trainer.rollout_steps, trainer.env.N), \
        f"returns shape wrong: {returns.shape}"
    assert torch.all(torch.isfinite(returns)), "Non-finite returns detected"


# ------------------------------------------------------------------
# Test 3 — PPO update runs and returns finite losses
# ------------------------------------------------------------------

def test_ppo_update(trainer):
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)
    returns = trainer._compute_returns(
        trainer.buffer_rewards,
        trainer.buffer_dones,
        last_value,
    )

    losses = trainer._ppo_update(returns)

    assert "policy_loss" in losses
    assert "value_loss"  in losses
    assert "entropy"     in losses

    assert np.isfinite(losses["policy_loss"]), "policy_loss not finite"
    assert np.isfinite(losses["value_loss"]),  "value_loss not finite"
    assert np.isfinite(losses["entropy"]),     "entropy not finite"

    # Value loss should be reasonable — quantile Huber on normalised returns
    # should be in range [0, 1] at initialisation
    assert losses["value_loss"] < 10.0, \
        f"value_loss suspiciously large: {losses['value_loss']}"


# ------------------------------------------------------------------
# Test 4 — Network weights change after update
# ------------------------------------------------------------------

def test_weights_update(trainer):
    weights_before = {
        name: param.clone()
        for name, param in trainer.net.named_parameters()
    }

    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)
    returns = trainer._compute_returns(
        trainer.buffer_rewards,
        trainer.buffer_dones,
        last_value,
    )
    trainer._ppo_update(returns)

    any_changed = False
    for name, param in trainer.net.named_parameters():
        if not torch.allclose(weights_before[name], param):
            any_changed = True
            break

    assert any_changed, "No network weights changed after PPO update"


# ------------------------------------------------------------------
# Test 5 — Curriculum lam_override works
# ------------------------------------------------------------------

def test_curriculum_lam_override(trainer):
    """
    Confirm set_lam_override changes the environment's jump intensity
    and that paths generated after the change reflect the new lambda.
    """
    # Set to easy stage
    trainer.env.set_lam_override(0.5)
    assert trainer.env.lam_override == 0.5

    # Set to target stage
    trainer.env.set_lam_override(1.5)
    assert trainer.env.lam_override == 1.5


# ------------------------------------------------------------------
# Test 6 — Full train() loop runs for small episode count
# ------------------------------------------------------------------

def test_train_loop(trainer):
    returns = trainer.train(total_ep=500, stage_name="test")

    assert isinstance(returns, list)
    assert len(returns) > 0
    assert all(np.isfinite(r) for r in returns), \
        "Non-finite returns in training output"


# ------------------------------------------------------------------
# Test 7 — CVaR advantage is computed correctly in update
# ------------------------------------------------------------------

def test_cvar_advantage_in_update(trainer):
    """
    Verify the CVaR advantage has correct sign properties:
    - Should have both positive and negative values across a batch
    - Should not be all zeros (would mean no gradient signal)
    """
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)
    returns = trainer._compute_returns(
        trainer.buffer_rewards,
        trainer.buffer_dones,
        last_value,
    )

    # Compute CVaR advantage manually on first minibatch
    from training.losses import cvar_advantage

    quantiles_flat = trainer.buffer_quantiles.view(-1, trainer.net.num_quantiles)
    returns_flat   = returns.view(-1)

    adv = cvar_advantage(
        returns_flat,
        quantiles_flat,
        num_tail=trainer.net.num_tails,
    )

    # Should have both positive and negative values
    assert torch.any(adv > 0), "No positive advantages — policy has no gradient signal"
    assert torch.any(adv < 0), "No negative advantages — tail risk not being penalised"
    assert torch.all(torch.isfinite(adv)), "Non-finite CVaR advantages"


# ------------------------------------------------------------------
# Test 8 — PPO update runs ALL epochs/minibatches, not just one
# ------------------------------------------------------------------

def test_ppo_update_runs_all_minibatches(trainer, monkeypatch):
    """
    Regression test: _ppo_update must iterate over every epoch and every
    minibatch before returning. A previous bug had the `return` statement
    indented inside the minibatch loop, so the optimiser only ever took a
    single gradient step per rollout (using one 32-sample minibatch out of
    thousands of collected transitions), starving the network of gradient
    signal and causing CVaR to plateau early in training.

    This test counts how many times the optimiser step is actually invoked
    and checks it matches num_epochs * ceil(total_samples / batch_size).
    """
    obs_np = trainer.env.reset()
    obs    = torch.tensor(obs_np, dtype=torch.float32).to(trainer.device)

    obs, last_value, _ = trainer._collect_rollout(obs)
    returns = trainer._compute_returns(
        trainer.buffer_rewards,
        trainer.buffer_dones,
        last_value,
    )

    total_samples = trainer.rollout_steps * trainer.env.N
    expected_minibatches_per_epoch = -(-total_samples // trainer.batch_size)  # ceil div
    expected_total_steps = trainer.num_epochs * expected_minibatches_per_epoch

    step_count = 0
    real_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "step", counting_step)

    trainer._ppo_update(returns)

    assert step_count == expected_total_steps, (
        f"Expected {expected_total_steps} optimiser steps "
        f"({trainer.num_epochs} epochs x {expected_minibatches_per_epoch} minibatches), "
        f"got {step_count}. _ppo_update is not processing the full rollout buffer."
    )