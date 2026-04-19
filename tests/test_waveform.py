"""Tests for smbhb_inspiral.waveform — strain and characteristic strain."""

from __future__ import annotations

import numpy as np
import pytest

from smbhb_inspiral.physics import integrate_inspiral
from smbhb_inspiral.waveform import (
    characteristic_strain_analytic,
    characteristic_strain_track,
    strain_cross,
    strain_plus,
)

# ---------------------------------------------------------------------------
# Reference system parameters
# ---------------------------------------------------------------------------

_M1_REF = 5e8    # M_sun
_M2_REF = 2e8    # M_sun
_F0_REF = 3e-9   # Hz
_MC_REF = 2.6976e8   # M_sun  (verified chirp mass)
_DL_REF = 500.0  # Mpc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_trajectory():
    """Full inspiral trajectory for the reference system."""
    return integrate_inspiral(_M1_REF, _M2_REF, _F0_REF)


# ---------------------------------------------------------------------------
# strain_plus / strain_cross
# ---------------------------------------------------------------------------

def test_strain_plus_shape(reference_trajectory) -> None:
    """strain_plus output length must match trajectory length."""
    hp = strain_plus(reference_trajectory, _DL_REF)
    assert hp.shape == reference_trajectory.f_gw.shape


def test_strain_cross_shape(reference_trajectory) -> None:
    """strain_cross output length must match trajectory length."""
    hx = strain_cross(reference_trajectory, _DL_REF)
    assert hx.shape == reference_trajectory.f_gw.shape


def test_strain_face_on_equal_amplitude_envelope(reference_trajectory) -> None:
    """At iota=0 the amplitude envelope is the same for both polarizations.

    For a face-on binary (iota=0) the inclination factors are:
      h+  ∝ (1 + cos²0)/2 * cos(Φ) = cos(Φ)
      h×  ∝ cos(0) * sin(Φ)         = sin(Φ)

    Because cos and sin share the same amplitude, the Pythagorean identity
    gives h+² + h×² = A(t)²  where A(t) is the slowly-varying amplitude
    prefactor.  We verify this identity against an independently computed
    amplitude to better than 0.1% relative error.
    """
    from smbhb_inspiral.waveform import _amplitude_prefactor
    from smbhb_inspiral.constants import M_SUN, MPC

    hp = strain_plus(reference_trajectory, _DL_REF, iota=0.0)
    hx = strain_cross(reference_trajectory, _DL_REF, iota=0.0)

    # Pythagorean envelope
    envelope_sq = hp**2 + hx**2

    # Independent amplitude from the raw prefactor
    mc_kg = reference_trajectory.chirp_mass_msun * M_SUN
    dl_m = _DL_REF * MPC
    A_sq = _amplitude_prefactor(mc_kg, reference_trajectory.f_gw, dl_m) ** 2

    # h+² + h×² must equal A² everywhere to within 0.1%
    np.testing.assert_allclose(envelope_sq, A_sq, rtol=1e-3)


def test_strain_plus_finite_values(reference_trajectory) -> None:
    """strain_plus must produce finite (non-NaN, non-inf) values."""
    hp = strain_plus(reference_trajectory, _DL_REF)
    assert np.all(np.isfinite(hp))


def test_strain_cross_finite_values(reference_trajectory) -> None:
    """strain_cross must produce finite (non-NaN, non-inf) values."""
    hx = strain_cross(reference_trajectory, _DL_REF)
    assert np.all(np.isfinite(hx))


def test_strain_cross_negligible_at_edge_on(reference_trajectory) -> None:
    """h_cross must be negligibly small for an edge-on system (iota = pi/2).

    The inclination factor is cos(iota).  At iota = pi/2, numpy's floating-
    point cos is ~6.1e-17 (machine epsilon), not exactly zero.  The cross-
    polarization amplitude is therefore ~6e-17 times the face-on amplitude,
    which is well below any astrophysical threshold.

    We verify that max(|h×|) at edge-on is less than 1e-14 of the face-on
    amplitude, confirming the cos(iota) suppression is working correctly.
    """
    hp_face_on = strain_plus(reference_trajectory, _DL_REF, iota=0.0)
    hx_edge_on = strain_cross(reference_trajectory, _DL_REF, iota=np.pi / 2.0)

    face_on_amplitude = np.max(np.abs(hp_face_on))
    edge_on_amplitude = np.max(np.abs(hx_edge_on))

    # Ratio should be ~cos(pi/2) ≈ 6e-17 — far below 1e-14
    assert edge_on_amplitude < 1e-14 * face_on_amplitude


# ---------------------------------------------------------------------------
# characteristic_strain_analytic
# ---------------------------------------------------------------------------

def test_hc_analytic_reference() -> None:
    """Verified value: h_c(1e-8 Hz, M_c=2.6976e8 Msun, d_L=500 Mpc) ≈ 6.39e-14.

    The package uses the sky-, polarization-, and inclination-averaged SPA
    formula (Moore, Cole & Berry 2014, Eq. 14; Sesana et al. 2008, Eq. 2),
    normalized so that SNR^2 = integral h_c^2 / h_n^2 d(ln f) with
    h_n^2 = f S_n(f).
    """
    hc = characteristic_strain_analytic(
        f_hz=np.array([1e-8]),
        chirp_mass_msun=_MC_REF,
        d_l_mpc=_DL_REF,
    )
    assert hc[0] == pytest.approx(6.39e-14, rel=0.05)


def test_hc_scales_inversely_with_distance() -> None:
    """h_c must halve when the luminosity distance doubles."""
    f = np.array([1e-8])
    hc_ref = characteristic_strain_analytic(f, _MC_REF, _DL_REF)
    hc_double = characteristic_strain_analytic(f, _MC_REF, 2.0 * _DL_REF)
    assert hc_ref[0] / hc_double[0] == pytest.approx(2.0, rel=1e-10)


def test_hc_track_matches_analytic(reference_trajectory) -> None:
    """characteristic_strain_track must agree with characteristic_strain_analytic
    to machine precision (the track function delegates to the analytic one)."""
    f_track, hc_track = characteristic_strain_track(reference_trajectory, _DL_REF)
    hc_analytic = characteristic_strain_analytic(
        f_track, reference_trajectory.chirp_mass_msun, _DL_REF
    )
    np.testing.assert_allclose(hc_track, hc_analytic, rtol=1e-12)


def test_hc_power_law_slope() -> None:
    """h_c(f) ∝ f^(-1/6): the log-log slope must equal -1/6 to 1%."""
    # Use a wide frequency range spanning four decades
    freqs = np.geomspace(1e-10, 1e-6, num=100)
    hc = characteristic_strain_analytic(freqs, _MC_REF, _DL_REF)

    log_f = np.log(freqs)
    log_hc = np.log(hc)

    # Linear fit to log-log data
    slope, _ = np.polyfit(log_f, log_hc, 1)

    assert slope == pytest.approx(-1.0 / 6.0, rel=0.01)


def test_hc_analytic_output_shape() -> None:
    """Output shape must match input frequency array shape."""
    freqs = np.geomspace(1e-10, 1e-6, num=50)
    hc = characteristic_strain_analytic(freqs, _MC_REF, _DL_REF)
    assert hc.shape == freqs.shape


def test_hc_analytic_positive_values() -> None:
    """Characteristic strain must be strictly positive."""
    freqs = np.geomspace(1e-10, 1e-6, num=50)
    hc = characteristic_strain_analytic(freqs, _MC_REF, _DL_REF)
    assert np.all(hc > 0.0)
