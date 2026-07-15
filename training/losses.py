import torch
import torch.nn.functional as F


def quantile_huber_loss(
    quantiles: torch.Tensor,   # (batch, N) predicted quantile estimates theta_i(s)
    returns: torch.Tensor,     # (batch,)   target returns R_t
    tau: torch.Tensor,         # (N,)       quantile levels tau_i
    kappa: float = 1.0,        # Huber loss threshold
) -> torch.Tensor:
    """
    Quantile Huber loss for distributional RL (Dabney et al. 2018).

    For each quantile estimate theta_i(s), computes an asymmetrically
    weighted Huber loss that pushes theta_i toward the tau_i-th quantile
    of the return distribution.

    The asymmetric weight |tau_i - 1_{u_i < 0}| ensures:
        - When u_i > 0 (return > estimate): weight = tau_i
          → underestimation penalised proportionally to tau_i
        - When u_i < 0 (return < estimate): weight = 1 - tau_i
          → overestimation penalised proportionally to 1 - tau_i

    For tail quantiles (small tau_i), overestimation is penalised heavily
    — the loss pushes low quantiles to accurately capture worst outcomes.

    Args:
        quantiles: (batch, N) — critic output theta_i(s_t) for i=1,...,N
        returns:   (batch,)   — Monte Carlo or GAE return targets R_t
        tau:       (N,)       — fixed quantile levels from register_buffer
        kappa:     float      — Huber loss threshold (1.0 per dissertation spec)

    Returns:
        scalar loss averaged over batch and quantiles
    """
    batch_size = quantiles.shape[0]
    N          = quantiles.shape[1]

    # Expand for pairwise computation:
    # returns:   (batch, 1, 1)  → broadcasts to (batch, N, N)
    # quantiles: (batch, 1, N)  → broadcasts to (batch, N, N)
    # Each entry [b, i, j] = R_t[b] - theta_j(s)[b]
    # We compute TD errors for all (target, quantile) pairs
    returns_expanded   = returns.unsqueeze(-1).unsqueeze(-1)   # (batch, 1, 1)
    quantiles_expanded = quantiles.unsqueeze(1)                # (batch, 1, N)

    # TD error: u_{ij} = R_t - theta_j(s)
    # Shape: (batch, N, N) — N target copies × N quantile estimates
    # We use N target copies so each quantile gets gradient from all targets
    # In practice with single returns, dim 1 is trivially size 1 then broadcast
    u = returns_expanded - quantiles_expanded   # (batch, 1, N)

    # Huber loss L_kappa(u):
    #   |u| <= kappa: 0.5 * u^2 / kappa
    #   |u| >  kappa: |u| - 0.5 * kappa
    # F.huber_loss with delta=kappa gives this form
    huber = F.huber_loss(
        quantiles_expanded.expand_as(u),
        returns_expanded.expand_as(u),
        reduction="none",
        delta=kappa,
    )   # (batch, 1, N)

    # Asymmetric weight: |tau_i - 1_{u < 0}|
    # tau shape: (N,) → expand to (1, 1, N) for broadcasting
    tau_expanded = tau.view(1, 1, N)            # (1, 1, N)

    # Indicator: 1 where TD error is negative (return < estimate)
    indicator = (u < 0).float()                 # (batch, 1, N)

    # Asymmetric weight — core of quantile regression
    weight = (tau_expanded - indicator).abs()   # (batch, 1, N)

    # Weighted Huber loss
    loss = weight * huber                       # (batch, 1, N)

    # Mean over batch and quantiles — scalar loss
    return loss.mean()


def cvar_advantage(
    returns: torch.Tensor,     # (batch,)   GAE returns R_t
    quantiles: torch.Tensor,   # (batch, N) quantile estimates theta_i(s_t)
    num_tail: int,               # number of tail quantiles = floor(alpha * N)
) -> torch.Tensor:
    """
    CVaR advantage for Phase 3 policy gradient.

    Replaces standard GAE advantage with a tail-risk-aware alternative:

        A_t^CVaR = R_t - CVaR_alpha(Z_phi(s_t))
                 = R_t - mean of n_tail lowest quantiles of Z_phi(s_t)

    When R_t falls below CVaR(s_t), the advantage is negative — the
    policy is penalised for actions that led to tail outcomes.
    When R_t is above CVaR(s_t), the advantage is positive — the
    policy is rewarded for avoiding the tail.

    This creates a gradient that specifically pushes the policy away
    from states and actions that produce tail losses, rather than
    simply maximising expected return.

    Args:
        returns:   (batch,) GAE return targets
        quantiles: (batch, N) distributional critic output
        n_tail:    number of quantiles in CVaR tail

    Returns:
        (batch,) CVaR advantages
    """
    # Sort quantiles ascending — lowest = worst outcomes = left tail
    sorted_q, _ = torch.sort(quantiles, dim=-1)   # (batch, N)

    # CVaR = mean of num_tail lowest quantiles
    cvar = sorted_q[:, :num_tail].mean(dim=-1)       # (batch,)

    # Advantage = return - CVaR baseline
    # Positive when return > CVaR (good outcome relative to tail)
    # Negative when return < CVaR (tail outcome, penalise the policy)
    advantage = returns - cvar                      # (batch,)

    return advantage