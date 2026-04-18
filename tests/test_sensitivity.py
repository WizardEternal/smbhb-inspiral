"""Tests for smbhb_inspiral.sensitivity — LISA and NANOGrav sensitivity curves.

Covers output shape, physical plausibility, reference values, and interpolation
behaviour for both detector families.
"""

from __future__ import annotations

import numpy as np
import pytest

from smbhb_inspiral.sensitivity import (
    lisa_sensitivity_hc,
    nanograv_15yr_sensitivity_hc,
    nanograv_15yr_sensitivity_hc_interp,
)

# ---------------------------------------------------------------------------
# LISA sensitivity curve — lisa_sensitivity_hc
# ---------------------------------------------------------------------------

# Frequency grid used across multiple LISA tests (1e-5 to 1 Hz, 200 points).
_LISA_F = np.logspace(-5, 0, 200)


def test_lisa_output_shape() -> None:
    """Input array of N frequencies produces output array of the same shape."""
    f = np.logspace(-4, -1, 57)
    h_c = lisa_sensitivity_hc(f)
    assert h_c.shape == f.shape


def test_lisa_positive_values() -> None:
    """All h_c values must be strictly positive over the valid frequency range."""
    h_c = lisa_sensitivity_hc(_LISA_F)
    assert np.all(h_c > 0.0)


def test_lisa_finite_values() -> None:
    """No NaN or infinite values should appear in the valid range (1e-5 to 1 Hz)."""
    h_c = lisa_sensitivity_hc(_LISA_F)
    assert np.all(np.isfinite(h_c))


def test_lisa_reference_value() -> None:
    """h_c at 1 mHz should be approximately 3.75e-21 (Robson+2019) to 20% tolerance."""
    f_ref = np.array([1.0e-3])
    h_c = lisa_sensitivity_hc(f_ref)
    assert h_c[0] == pytest.approx(3.75e-21, rel=0.20)


def test_lisa_best_sensitivity_near_few_mhz() -> None:
    """The minimum h_c (best sensitivity) should occur between 1 mHz and 10 mHz."""
    f_fine = np.logspace(-3, -2, 500)       # 1 mHz to 10 mHz
    h_c_fine = lisa_sensitivity_hc(f_fine)
    f_outside = np.logspace(-5, -3, 200)    # below 1 mHz
    h_c_outside = lisa_sensitivity_hc(f_outside)

    best_in_window = h_c_fine.min()
    # The global minimum over 1e-5 to 1 Hz must lie inside the 1–10 mHz window.
    assert best_in_window <= h_c_outside.min()


def test_lisa_sensitivity_rises_at_edges() -> None:
    """LISA has a bathtub shape: h_c at the edges must exceed the trough value.

    h_c at 1e-5 Hz (low-f noise wall) and at 0.5 Hz (high-f noise wall) must
    both be larger than h_c at 3 mHz (near the trough minimum).
    """
    f_low = np.array([1.0e-5])
    f_trough = np.array([3.0e-3])
    f_high = np.array([0.5])

    hc_low = lisa_sensitivity_hc(f_low)[0]
    hc_trough = lisa_sensitivity_hc(f_trough)[0]
    hc_high = lisa_sensitivity_hc(f_high)[0]

    assert hc_low > hc_trough
    assert hc_high > hc_trough


# ---------------------------------------------------------------------------
# NANOGrav raw data — nanograv_15yr_sensitivity_hc
# ---------------------------------------------------------------------------


def test_nanograv_returns_tuple() -> None:
    """nanograv_15yr_sensitivity_hc() must return a tuple of two arrays."""
    result = nanograv_15yr_sensitivity_hc()
    assert isinstance(result, tuple)
    assert len(result) == 2
    f_hz, h_c = result
    assert isinstance(f_hz, np.ndarray)
    assert isinstance(h_c, np.ndarray)


def test_nanograv_array_lengths() -> None:
    """Both frequency and h_c arrays must have exactly 11 elements."""
    f_hz, h_c = nanograv_15yr_sensitivity_hc()
    assert len(f_hz) == 11
    assert len(h_c) == 11


def test_nanograv_frequency_range() -> None:
    """NANOGrav frequencies must span approximately 1e-9 to 1e-7 Hz."""
    f_hz, _ = nanograv_15yr_sensitivity_hc()
    assert f_hz.min() >= 1.0e-10       # not below 0.1 nHz
    assert f_hz.min() <= 1.0e-8        # not above 10 nHz
    assert f_hz.max() >= 1.0e-8        # not below 10 nHz
    assert f_hz.max() <= 1.0e-6        # not above 1 µHz


def test_nanograv_hc_range() -> None:
    """All NANOGrav h_c sensitivity values must lie between 1e-15 and 1e-11."""
    _, h_c = nanograv_15yr_sensitivity_hc()
    assert np.all(h_c >= 1.0e-15)
    assert np.all(h_c <= 1.0e-11)


# ---------------------------------------------------------------------------
# NANOGrav interpolated curve — nanograv_15yr_sensitivity_hc_interp
# ---------------------------------------------------------------------------


def test_nanograv_interp_reference_values() -> None:
    """Interpolated sensitivity at 5 nHz ≈ 8e-15 and at 10 nHz ≈ 4e-15 (30% tol)."""
    f_test = np.array([5.0e-9, 1.0e-8])
    h_c = nanograv_15yr_sensitivity_hc_interp(f_test)
    assert h_c[0] == pytest.approx(8.0e-15, rel=0.30)
    assert h_c[1] == pytest.approx(4.0e-15, rel=0.30)


def test_nanograv_interp_outside_range_returns_inf() -> None:
    """Frequencies outside the digitized range (1e-10 Hz and 1e-6 Hz) must return inf."""
    f_out = np.array([1.0e-10, 1.0e-6])
    h_c = nanograv_15yr_sensitivity_hc_interp(f_out)
    assert np.all(np.isinf(h_c))


def test_nanograv_interp_monotonic_in_minimum_region() -> None:
    """Between the best-sensitivity point and the high-frequency end, h_c should rise.

    The PI sensitivity curve has a minimum somewhere in the band.  From that
    minimum to the highest frequency in the digitized range the curve must be
    non-decreasing (sensitivity gets worse at higher frequencies because fewer
    pulsar-timing harmonics contribute).
    """
    f_hz, h_c_raw = nanograv_15yr_sensitivity_hc()

    # Find the index of the minimum sensitivity in the raw data.
    idx_min = int(np.argmin(h_c_raw))

    # Interpolate a dense grid from the minimum frequency onwards.
    f_right = np.logspace(
        np.log10(f_hz[idx_min]),
        np.log10(f_hz[-1]),
        200,
    )
    h_c_right = nanograv_15yr_sensitivity_hc_interp(f_right)

    # h_c must be non-decreasing (allow tiny numerical noise of 1%).
    diffs = np.diff(h_c_right)
    assert np.all(diffs >= -1e-2 * h_c_right[:-1])
