"""End-to-end smoke tests for the full smbhb_inspiral pipeline.

Runs the complete workflow for the canonical reference system:
  integrate_inspiral → characteristic_strain_track → classify_system

Verifies that all outputs have expected shapes, physical units, and
mutually consistent values.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from smbhb_inspiral.em_detectability import (
    classify_system,
)
from smbhb_inspiral.physics import (
    InspiralTrajectory,
    chirp_mass,
    f_isco,
    integrate_inspiral,
)
from smbhb_inspiral.waveform import (
    characteristic_strain_analytic,
    characteristic_strain_track,
)

# ---------------------------------------------------------------------------
# Reference system
# ---------------------------------------------------------------------------

_M1 = 5e8    # M_sun
_M2 = 2e8    # M_sun
_F0 = 3e-9   # Hz  (initial GW frequency)
_DL = 500.0  # Mpc  (luminosity distance)
_Z  = 0.1    # redshift


@pytest.fixture(scope="module")
def pipeline_outputs():
    """Run the full pipeline once; return a namespace with all results."""
    # Step 1: integrate the inspiral
    traj = integrate_inspiral(_M1, _M2, _F0)

    # Step 2: characteristic strain track
    f_track, hc_track = characteristic_strain_track(traj, _DL)

    # Step 3: analytic characteristic strain on the same frequency grid
    mc = traj.chirp_mass_msun
    hc_analytic = characteristic_strain_analytic(f_track, mc, _DL)

    # Step 4: EM detectability at the initial frequency
    em_result = classify_system(
        m_total_msun=_M1 + _M2,
        f_gw_hz=_F0,
        z=_Z,
    )

    return types.SimpleNamespace(
        traj=traj,
        f_track=f_track,
        hc_track=hc_track,
        hc_analytic=hc_analytic,
        mc=mc,
        em_result=em_result,
    )


# ---------------------------------------------------------------------------
# Shape / size checks
# ---------------------------------------------------------------------------

def test_trajectory_length(pipeline_outputs) -> None:
    """Inspiral trajectory must contain 10 001 points (verified reference)."""
    assert len(pipeline_outputs.traj.t) == 10_001


def test_frequency_array_length(pipeline_outputs) -> None:
    """Characteristic strain frequency array must match trajectory length."""
    assert pipeline_outputs.f_track.shape == (10_001,)


def test_hc_track_length(pipeline_outputs) -> None:
    """Characteristic strain values array must match trajectory length."""
    assert pipeline_outputs.hc_track.shape == (10_001,)


# ---------------------------------------------------------------------------
# Physical value checks — trajectory
# ---------------------------------------------------------------------------

def test_initial_gw_frequency(pipeline_outputs) -> None:
    """First GW frequency point must equal f0 (initial condition)."""
    assert pipeline_outputs.traj.f_gw[0] == pytest.approx(_F0, rel=1e-6)


def test_final_gw_frequency_near_isco(pipeline_outputs) -> None:
    """Final GW frequency must be within 5% of f_isco(M_tot)."""
    traj = pipeline_outputs.traj
    expected_isco = f_isco(traj.total_mass_msun)
    assert traj.f_gw[-1] == pytest.approx(expected_isco, rel=0.05)


def test_chirp_mass_stored_correctly(pipeline_outputs) -> None:
    """Trajectory's chirp_mass_msun must match the stand-alone chirp_mass()."""
    assert pipeline_outputs.traj.chirp_mass_msun == pytest.approx(
        chirp_mass(_M1, _M2), rel=1e-10
    )


def test_total_mass_stored_correctly(pipeline_outputs) -> None:
    """Trajectory's total_mass_msun must equal m1 + m2."""
    assert pipeline_outputs.traj.total_mass_msun == pytest.approx(
        _M1 + _M2, rel=1e-10
    )


def test_trajectory_all_finite(pipeline_outputs) -> None:
    """All trajectory arrays must be finite (no NaN / inf)."""
    traj = pipeline_outputs.traj
    for arr_name in ("t", "f_gw", "f_orb", "a", "v_over_c", "phi"):
        arr = getattr(traj, arr_name)
        assert np.all(np.isfinite(arr)), f"Non-finite values in trajectory.{arr_name}"


# ---------------------------------------------------------------------------
# Physical value checks — characteristic strain
# ---------------------------------------------------------------------------

def test_hc_track_matches_analytic_exactly(pipeline_outputs) -> None:
    """characteristic_strain_track must equal characteristic_strain_analytic
    to machine precision (it delegates to the same formula)."""
    np.testing.assert_allclose(
        pipeline_outputs.hc_track,
        pipeline_outputs.hc_analytic,
        rtol=1e-12,
    )


def test_hc_monotonically_increasing_with_frequency(pipeline_outputs) -> None:
    """h_c ∝ f^(-1/6) decreases with frequency, so lower-f → higher h_c.

    The track runs from low to high frequency, meaning hc_track should
    be monotonically decreasing.
    """
    diffs = np.diff(pipeline_outputs.hc_track)
    assert np.all(diffs <= 0.0)


def test_hc_at_reference_frequency(pipeline_outputs) -> None:
    """h_c near 1e-8 Hz should be in the ballpark of the verified reference value."""
    # Find the point closest to 1e-8 Hz in the track
    idx = np.argmin(np.abs(pipeline_outputs.f_track - 1e-8))
    hc_val = pipeline_outputs.hc_track[idx]
    # Verified: ~6.39e-14 at exactly 1e-8 Hz; nearby values will be close
    assert hc_val == pytest.approx(6.39e-14, rel=0.20)   # generous tolerance for off-grid


def test_hc_positive_everywhere(pipeline_outputs) -> None:
    """Characteristic strain must be strictly positive throughout the track."""
    assert np.all(pipeline_outputs.hc_track > 0.0)


# ---------------------------------------------------------------------------
# Physical value checks — EM detectability
# ---------------------------------------------------------------------------

def test_em_result_type(pipeline_outputs) -> None:
    """classify_system must return an EMDetectabilityResult."""
    from smbhb_inspiral.em_detectability import EMDetectabilityResult
    assert isinstance(pipeline_outputs.em_result, EMDetectabilityResult)


def test_em_p_obs_consistent_with_f0() -> None:
    """P_obs from classify_system must agree with the hand-computed value.

    P_rest = 2/f0;  P_obs = P_rest * (1 + z)  [seconds]
    convert to days and compare.
    """
    result = classify_system(_M1 + _M2, _F0, _Z)
    p_rest_s = 2.0 / _F0
    p_obs_expected_days = p_rest_s * (1.0 + _Z) / 86_400.0
    assert result.p_obs_days == pytest.approx(p_obs_expected_days, rel=1e-8)


def test_em_reference_not_in_stripe82(pipeline_outputs) -> None:
    """Reference system at f0=3e-9 Hz (P_obs ≈ 8488 days) is outside Stripe 82."""
    assert pipeline_outputs.em_result.in_stripe82_window is False


def test_em_recovery_fractions_all_zero_for_reference(pipeline_outputs) -> None:
    """All recovery fractions must be 0 for the out-of-window reference system."""
    em = pipeline_outputs.em_result
    for field in (
        "recovery_sinusoidal_stripe82",
        "recovery_sawtooth_stripe82",
        "recovery_sinusoidal_ptf",
        "recovery_sawtooth_ptf",
        "recovery_sinusoidal_lsst",
        "recovery_sawtooth_lsst",
    ):
        assert getattr(em, field) == 0.0, f"{field} should be 0 but got {getattr(em, field)}"
