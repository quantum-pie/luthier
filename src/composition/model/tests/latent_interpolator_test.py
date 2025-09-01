import math
import sys
import torch
import pytest

from src.composition.model.latent_interpolator import LatentInterpolator


def test_initialization_and_constant_before_change():
    li = LatentInterpolator(transition_duration=4, smooth=False)
    z_a = torch.ones(5)

    assert not li.is_initialized()
    li.update(z_a, current_bar_pos=0.0)
    assert li.is_initialized()

    # Before any change, z stays constant regardless of bar position
    z0 = li.get_current_z(torch.tensor(0.0))
    z1 = li.get_current_z(torch.tensor(1.0))
    z2 = li.get_current_z(torch.tensor(2.0))
    assert torch.allclose(z0, z_a)
    assert torch.allclose(z1, z_a)
    assert torch.allclose(z2, z_a)


def test_linear_interpolation_and_finalize():
    li = LatentInterpolator(transition_duration=4, smooth=False)
    z_a = torch.ones(3)
    z_b = torch.zeros(3)
    li.update(z_a, current_bar_pos=torch.tensor(0.0))

    # Start transition at bar 2
    li.update(z_b, current_bar_pos=torch.tensor(2.0))

    # At the same bar, alpha = 0 => still old z
    z_at_2 = li.get_current_z(torch.tensor(2.0))
    assert torch.allclose(z_at_2, z_a)

    # After 1 bar, alpha = (3-2)/4 = 0.25 => 0.75*z_a + 0.25*z_b
    z_at_3 = li.get_current_z(torch.tensor(3.0))
    assert torch.allclose(z_at_3, 0.75 * z_a + 0.25 * z_b, atol=1e-6)

    # After duration bars, alpha >= 1 => finalized at z_b and stays there
    z_at_6 = li.get_current_z(torch.tensor(6.0))
    z_at_7 = li.get_current_z(torch.tensor(7.0))
    assert torch.allclose(z_at_6, z_b, atol=1e-6)
    assert torch.allclose(z_at_7, z_b, atol=1e-6)


def test_smooth_cosine_interpolation_factor():
    li = LatentInterpolator(transition_duration=4, smooth=True)
    z_a = torch.zeros(2)
    z_b = torch.ones(2)
    li.update(z_a, current_bar_pos=torch.tensor(0.0))
    li.update(z_b, current_bar_pos=torch.tensor(10.0))

    # Linear alpha at bar 11 is 0.25, smooth alpha = 0.5*(1 - cos(pi*0.25))
    linear_alpha = (11.0 - 10.0) / 4.0
    smooth_alpha = 0.5 * (1.0 - math.cos(math.pi * linear_alpha))

    z_at_11 = li.get_current_z(torch.tensor(11.0))
    expected = (1 - smooth_alpha) * z_a + smooth_alpha * z_b
    assert torch.allclose(z_at_11, expected, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main())
