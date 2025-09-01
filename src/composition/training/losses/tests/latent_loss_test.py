import sys
import torch
import pytest

from src.composition.training.losses.latent_loss import kl_loss


def test_kl_loss_zero_when_distributions_identical():
    B, D = 4, 7
    mu = torch.randn(B, D)
    logvar = torch.randn(B, D)  # allow any variance

    out = kl_loss(mu, logvar, mu, logvar)
    assert out.shape == (B,)
    # Identical Gaussians => KL = 0
    assert torch.allclose(out, torch.zeros(B), atol=1e-6)


def test_kl_loss_matches_closed_form_against_standard_normal_prior():
    # Prior p = N(0, I)
    # q = N(mu, diag(exp(logvar)))
    # KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    B, D = 3, 5
    mu_q = torch.randn(B, D)
    logvar_q = torch.randn(B, D)  # log variances

    mu_p = torch.zeros(B, D)
    logvar_p = torch.zeros(B, D)

    out = kl_loss(mu_q, logvar_q, mu_p, logvar_p)

    var_q = torch.exp(logvar_q)
    manual = 0.5 * (var_q + mu_q.pow(2) - 1.0 - logvar_q).sum(dim=-1)

    assert torch.allclose(out, manual, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main())
