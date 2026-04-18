"""Tests for smbhb_inspiral.em_detectability — EM survey recoverability.

Recovery fractions are locked to published values from
Lin, Charisi & Haiman 2026, ApJ 997, 316, Table 1 / Section 3.1.
Tests marked "locked" must not be changed without updating the citation.
"""

from __future__ import annotations

import pytest

from smbhb_inspiral.em_detectability import (
    RECOVERY_FRACTIONS,
    SURVEY_WINDOWS,
    EMDetectabilityResult,
    classify_system,
    in_survey_window,
    observer_frame_period,
    orbital_period_from_f_gw,
    recovery_fraction,
)

_SECONDS_PER_DAY = 86_400.0

# ---------------------------------------------------------------------------
# orbital_period_from_f_gw
# ---------------------------------------------------------------------------

def test_period_from_f_gw() -> None:
    """P_orb = 2 / f_gw because f_GW = 2 f_orb for a quasi-circular binary."""
    assert orbital_period_from_f_gw(1e-8) == pytest.approx(2e8, rel=1e-12)


def test_period_from_f_gw_negative_raises() -> None:
    """Negative or zero GW frequency must raise ValueError."""
    with pytest.raises(ValueError):
        orbital_period_from_f_gw(-1e-8)
    with pytest.raises(ValueError):
        orbital_period_from_f_gw(0.0)


# ---------------------------------------------------------------------------
# observer_frame_period
# ---------------------------------------------------------------------------

def test_observer_frame_period() -> None:
    """P_obs = P_rest * (1 + z); at z=1 the period doubles."""
    assert observer_frame_period(100.0, 1.0) == pytest.approx(200.0, rel=1e-12)


def test_observer_frame_period_zero_redshift() -> None:
    """At z=0 the observer-frame period equals the rest-frame period."""
    assert observer_frame_period(12345.6, 0.0) == pytest.approx(12345.6, rel=1e-12)


# ---------------------------------------------------------------------------
# in_survey_window  (Stripe 82: 200–1100 days)
# ---------------------------------------------------------------------------

def test_in_survey_window_stripe82_inside() -> None:
    """500 days is inside the Stripe 82 window [200, 1100]."""
    assert in_survey_window(500.0, "stripe82") is True


def test_in_survey_window_stripe82_too_short() -> None:
    """50 days is below the Stripe 82 minimum (200 days)."""
    assert in_survey_window(50.0, "stripe82") is False


def test_in_survey_window_stripe82_too_long() -> None:
    """2000 days exceeds the Stripe 82 maximum (1100 days)."""
    assert in_survey_window(2000.0, "stripe82") is False


def test_in_survey_window_stripe82_boundary_min() -> None:
    """Exactly at P_min (200 days) the system is inside the window."""
    p_min = SURVEY_WINDOWS["stripe82"]["P_min_days"]
    assert in_survey_window(p_min, "stripe82") is True


def test_in_survey_window_stripe82_boundary_max() -> None:
    """Exactly at P_max (1100 days) the system is inside the window."""
    p_max = SURVEY_WINDOWS["stripe82"]["P_max_days"]
    assert in_survey_window(p_max, "stripe82") is True


def test_in_survey_window_invalid_survey() -> None:
    """An unrecognised survey name must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown survey"):
        in_survey_window(500.0, "nonexistent_survey")


# ---------------------------------------------------------------------------
# RECOVERY_FRACTIONS — locked published numbers
# (Lin, Charisi & Haiman 2026, ApJ 997, 316, Table 1 / Section 3.1)
# ---------------------------------------------------------------------------

def test_recovery_fractions_locked_sinusoidal_ptf_like() -> None:
    """LOCKED: sinusoidal PTF-like recovery fraction = 0.45."""
    assert RECOVERY_FRACTIONS["sinusoidal"]["ptf_like"] == 0.45


def test_recovery_fractions_locked_sinusoidal_idealized() -> None:
    """LOCKED: sinusoidal idealized recovery fraction = 0.24."""
    assert RECOVERY_FRACTIONS["sinusoidal"]["idealized"] == 0.24


def test_recovery_fractions_locked_sinusoidal_lsst_like() -> None:
    """LOCKED: sinusoidal LSST-like recovery fraction = 0.23."""
    assert RECOVERY_FRACTIONS["sinusoidal"]["lsst_like"] == 0.23


def test_recovery_fractions_locked_sawtooth_ptf_like() -> None:
    """LOCKED: sawtooth PTF-like recovery fraction = 0.09."""
    assert RECOVERY_FRACTIONS["sawtooth"]["ptf_like"] == 0.09


def test_recovery_fractions_locked_sawtooth_idealized() -> None:
    """LOCKED: sawtooth idealized recovery fraction = 0.01."""
    assert RECOVERY_FRACTIONS["sawtooth"]["idealized"] == 0.01


def test_recovery_fractions_locked_sawtooth_lsst_like() -> None:
    """LOCKED: sawtooth LSST-like recovery fraction = 0.01."""
    assert RECOVERY_FRACTIONS["sawtooth"]["lsst_like"] == 0.01


# ---------------------------------------------------------------------------
# classify_system — reference system (outside Stripe82 window)
# ---------------------------------------------------------------------------

def test_classify_system_reference_p_obs() -> None:
    """classify_system(7e8, 3e-9, 0.1): P_obs ≈ 8488 days."""
    result = classify_system(7e8, 3e-9, 0.1)
    assert result.p_obs_days == pytest.approx(8488.0, rel=1e-3)


def test_classify_system_reference_not_in_stripe82() -> None:
    """Reference system is outside the Stripe 82 sensitivity window."""
    result = classify_system(7e8, 3e-9, 0.1)
    assert result.in_stripe82_window is False


def test_classify_system_reference_recovery_zero() -> None:
    """Reference system: all recovery fractions must be 0 (out of every window)."""
    result = classify_system(7e8, 3e-9, 0.1)
    assert result.recovery_sinusoidal_stripe82 == 0.0
    assert result.recovery_sawtooth_stripe82 == 0.0
    assert result.recovery_sinusoidal_ptf == 0.0
    assert result.recovery_sawtooth_ptf == 0.0
    assert result.recovery_sinusoidal_lsst == 0.0
    assert result.recovery_sawtooth_lsst == 0.0


# ---------------------------------------------------------------------------
# classify_system — in-window system
# ---------------------------------------------------------------------------

def test_classify_system_in_stripe82_window() -> None:
    """A system with P_obs = 500 days falls inside Stripe 82.

    f_gw = 2 / P_rest_s at z=0 so that P_obs = P_rest = 500 days.
    Recovery fractions should match the PTF-like lookup values.
    """
    target_days = 500.0
    p_rest_s = target_days * _SECONDS_PER_DAY
    f_gw_in = 2.0 / p_rest_s  # ≈ 4.63e-8 Hz

    result = classify_system(1e8, f_gw_in, z=0.0)

    assert result.in_stripe82_window is True
    assert result.p_obs_days == pytest.approx(500.0, rel=1e-8)

    # Recovery fractions come from RECOVERY_FRACTIONS via "ptf_like" cadence
    assert result.recovery_sinusoidal_stripe82 == pytest.approx(0.45, rel=1e-10)
    assert result.recovery_sawtooth_stripe82 == pytest.approx(0.09, rel=1e-10)


def test_classify_system_result_is_dataclass() -> None:
    """classify_system must return an EMDetectabilityResult instance."""
    result = classify_system(1e8, 1e-8, z=0.3)
    assert isinstance(result, EMDetectabilityResult)


def test_classify_system_recovery_fractions_in_range() -> None:
    """All recovery fractions must lie in [0, 1]."""
    result = classify_system(1e8, 1e-8, z=0.3)
    for field_name in (
        "recovery_sinusoidal_stripe82",
        "recovery_sawtooth_stripe82",
        "recovery_sinusoidal_ptf",
        "recovery_sawtooth_ptf",
        "recovery_sinusoidal_lsst",
        "recovery_sawtooth_lsst",
    ):
        val = getattr(result, field_name)
        assert 0.0 <= val <= 1.0, f"{field_name} = {val} is outside [0, 1]"


# ---------------------------------------------------------------------------
# classify_system — error handling
# ---------------------------------------------------------------------------

def test_classify_negative_redshift_raises() -> None:
    """Negative redshift must raise ValueError."""
    with pytest.raises(ValueError):
        classify_system(1e8, 1e-8, z=-0.1)


def test_classify_zero_frequency_raises() -> None:
    """Zero GW frequency must raise ValueError (unphysical)."""
    with pytest.raises(ValueError):
        classify_system(1e8, 0.0, z=0.1)


# ---------------------------------------------------------------------------
# recovery_fraction lookup consistency
# ---------------------------------------------------------------------------

def test_recovery_fraction_stripe82_sinusoidal() -> None:
    """recovery_fraction for stripe82/sinusoidal must match RECOVERY_FRACTIONS."""
    expected = RECOVERY_FRACTIONS["sinusoidal"][
        str(SURVEY_WINDOWS["stripe82"]["cadence"])
    ]
    assert recovery_fraction("sinusoidal", "stripe82") == expected


def test_recovery_fraction_invalid_survey_raises() -> None:
    """Invalid survey name must raise ValueError."""
    with pytest.raises(ValueError):
        recovery_fraction("sinusoidal", "mystery_survey")


def test_recovery_fraction_invalid_shape_raises() -> None:
    """Invalid signal shape must raise ValueError."""
    with pytest.raises(ValueError):
        recovery_fraction("triangle", "stripe82")  # type: ignore[arg-type]
