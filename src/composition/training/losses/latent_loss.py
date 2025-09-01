import torch


def kl_loss(mu_posterior, logvar_posterior, mu_prior, logvar_prior):
    """
    KL(q||p) where
      q = N(mu_posterior, diag(exp(logvar_posterior)))
      p = N(mu_prior,     diag(exp(logvar_prior)))

    Args:
        mu_posterior:   [B, D]
        logvar_posterior: [B, D]
        mu_prior:       [B, D]
        logvar_prior:   [B, D]
        reduction: "mean" | "sum" | "none"
    Returns:
        Scalar loss if reduction != "none", else per-sample KL [B]
    """
    # variances
    var_post = torch.exp(logvar_posterior)
    var_prior = torch.exp(logvar_prior)

    # KL(q||p) per-dimension
    # 0.5 * [ log(|Σ_p|/|Σ_q|) - D + tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) ]
    log_det_ratio = logvar_prior - logvar_posterior  # [B, D]
    trace_term = var_post / var_prior  # [B, D]
    mean_diff_sq = (mu_prior - mu_posterior).pow(2) / var_prior

    kl_per_dim = log_det_ratio + trace_term + mean_diff_sq - 1.0
    kl_per_sample = 0.5 * kl_per_dim.sum(dim=-1)  # [B]

    return kl_per_sample
