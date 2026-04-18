"""Tests for smbhb_inspiral.physics — PN inspiral mechanics.

Covers chirp mass, symmetric mass ratio, ISCO frequency, analytic merger time,
the ODE integrator, and the InspiralTrajectory dataclass.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from smbhb_inspiral.physics import (
    InspiralTrajectory,
    analytic_t_merge_circular,
    chirp_mass,
    f_isco,
    integrate_inspiral,
    symmetric_mass_ratio,
)

# ---------------------------------------------------------------------------
# Shared fixture — one expensive integration, reused across multiple tests
# ---------------------------------------------------------------------------

_M1_REF = 5e8   # M_sun
_M2_REF = 2e8   # M_sun
_F0_REF = 3e-9  # Hz


@pytest.fixture(scope="module")
def reference_trajectory() -> InspiralTrajectory:
    """Integrated inspiral for the canonical (5e8, 2e8 Msun) reference system."""
    return integrate_inspiral(_M1_REF, _M2_REF, _F0_REF)


@pytest.fixture(scope="module")
def reference_trajectory_0pn() -> InspiralTrajectory:
    """0PN inspiral for the canonical reference system."""
    return integrate_inspiral(_M1_REF, _M2_REF, _F0_REF, pn_order=0)


# ---------------------------------------------------------------------------
# chirp_mass
# ---------------------------------------------------------------------------

def test_chirp_mass_known_value() -> None:
    """Equal-mass binary: M_c = m / 2^(1/5).

    For m1 = m2 = m:
        M_c = (m·m)^(3/5) / (2m)^(1/5) = m^(6/5) / (2^(1/5) · m^(1/5))
            = m / 2^(1/5)

    Note: this is *not* m·(1/4)^(1/5); the equal-mass chirp mass equals
    the component mass divided by 2^(1/5) ≈ 1.1487, giving M_c ≈ 0.8706·m.
    """
    m = 10.0
    expected = m / 2.0 ** (1.0 / 5.0)   # ≈ 8.7055
    assert chirp_mass(m, m) == pytest.approx(expected, rel=1e-4)


def test_chirp_mass_reference_system() -> None:
    """Verified published value: chirp_mass(5e8, 2e8) ≈ 2.6976e8 Msun."""
    assert chirp_mass(5e8, 2e8) == pytest.approx(2.6976e8, rel=1e-3)


@given(
    m1=st.floats(min_value=1e6, max_value=1e10),
    m2=st.floats(min_value=1e6, max_value=1e10),
)
@settings(max_examples=200)
def test_chirp_mass_symmetry(m1: float, m2: float) -> None:
    """Chirp mass must be symmetric under label exchange."""
    assert chirp_mass(m1, m2) == pytest.approx(chirp_mass(m2, m1), rel=1e-12)


# ---------------------------------------------------------------------------
# symmetric_mass_ratio
# ---------------------------------------------------------------------------

@given(m=st.floats(min_value=1e4, max_value=1e12))
@settings(max_examples=100)
def test_symmetric_mass_ratio_equal_mass(m: float) -> None:
    """Equal-mass binary has eta = 0.25 exactly."""
    assert symmetric_mass_ratio(m, m) == pytest.approx(0.25, rel=1e-12)


@given(
    m1=st.floats(min_value=1e6, max_value=1e10),
    m2=st.floats(min_value=1e6, max_value=1e10),
)
@settings(max_examples=200)
def test_symmetric_mass_ratio_range(m1: float, m2: float) -> None:
    """Symmetric mass ratio must lie in (0, 0.25] for any positive masses."""
    eta = symmetric_mass_ratio(m1, m2)
    assert eta > 0.0
    assert eta <= 0.25 + 1e-12   # allow floating-point rounding at equal mass


# ---------------------------------------------------------------------------
# f_isco
# ---------------------------------------------------------------------------

def test_f_isco_reference() -> None:
    """Verified value: f_isco(7e8 Msun) ≈ 6.28e-6 Hz to 1%."""
    assert f_isco(7e8) == pytest.approx(6.28e-6, rel=0.01)


@given(m=st.floats(min_value=1e6, max_value=1e12))
@settings(max_examples=200)
def test_f_isco_scales_inversely_with_mass(m: float) -> None:
    """f_isco(M) * M must be constant (f_isco ∝ 1/M)."""
    reference_product = f_isco(1e8) * 1e8
    assert f_isco(m) * m == pytest.approx(reference_product, rel=1e-10)


# ---------------------------------------------------------------------------
# analytic_t_merge_circular
# ---------------------------------------------------------------------------

def test_analytic_t_merge_reference() -> None:
    """Verified published value: t_merge(5e8, 2e8, 3e-9) ≈ 3.07e14 s to 1%."""
    t = analytic_t_merge_circular(_M1_REF, _M2_REF, _F0_REF)
    assert t == pytest.approx(3.07e14, rel=0.01)


# ---------------------------------------------------------------------------
# integrate_inspiral — trajectory properties
# ---------------------------------------------------------------------------

def test_integrate_inspiral_reaches_isco(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """The last GW frequency should be close to f_isco of the total mass (within 5%)."""
    traj = reference_trajectory
    expected_f_isco = f_isco(traj.total_mass_msun)
    assert traj.f_gw[-1] == pytest.approx(expected_f_isco, rel=0.05)


def test_integrate_inspiral_monotonic_frequency(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """GW frequency must increase monotonically throughout the inspiral."""
    diffs = np.diff(reference_trajectory.f_gw)
    # Allow tiny floating-point noise but all steps must be non-negative
    assert np.all(diffs >= -1e-30)


def test_integrate_inspiral_monotonic_separation(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """Orbital separation must decrease monotonically as the binary inspirals."""
    diffs = np.diff(reference_trajectory.a)
    assert np.all(diffs <= 1e-3)   # allow tiny floating-point steps


def test_integrate_inspiral_output_length(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """Verified: the integrator returns exactly 10 001 output points."""
    assert len(reference_trajectory.t) == 10_001


def test_integrate_inspiral_0pn_matches_analytic(
    reference_trajectory_0pn: InspiralTrajectory,
) -> None:
    """At pn_order=0, integrated t_merge must agree with Peters formula to 1e-4."""
    t_numeric = reference_trajectory_0pn.t[-1]
    t_analytic = analytic_t_merge_circular(_M1_REF, _M2_REF, _F0_REF)
    assert t_numeric == pytest.approx(t_analytic, rel=1e-4)


def test_integrate_inspiral_1pn_differs_from_0pn(
    reference_trajectory: InspiralTrajectory,
    reference_trajectory_0pn: InspiralTrajectory,
) -> None:
    """1PN correction must change the merger time (should differ by >0.1%)."""
    t_0pn = reference_trajectory_0pn.t[-1]
    t_1pn = reference_trajectory.t[-1]
    relative_diff = abs(t_1pn - t_0pn) / t_0pn
    assert relative_diff > 1e-3   # 1PN shifts the merger time by ~0.4%


# ---------------------------------------------------------------------------
# InspiralTrajectory dataclass
# ---------------------------------------------------------------------------

def test_trajectory_dataclass_fields(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """All expected public fields must exist on InspiralTrajectory."""
    expected_fields = {
        "t",
        "f_gw",
        "f_orb",
        "a",
        "v_over_c",
        "phi",
        "chirp_mass_msun",
        "total_mass_msun",
        "eta",
        "pn_order",
    }
    actual_fields = {f.name for f in dataclasses.fields(reference_trajectory)}
    assert expected_fields <= actual_fields


def test_trajectory_initial_frequency(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """The first element of f_gw must match the initial frequency f0."""
    assert reference_trajectory.f_gw[0] == pytest.approx(_F0_REF, rel=1e-6)


def test_trajectory_f_orb_is_half_f_gw(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """Orbital frequency must equal f_gw / 2 everywhere."""
    np.testing.assert_allclose(
        reference_trajectory.f_orb,
        reference_trajectory.f_gw / 2.0,
        rtol=1e-12,
    )


def test_trajectory_v_over_c_range(
    reference_trajectory: InspiralTrajectory,
) -> None:
    """v/c should start near 0.025 and end near 0.32 for the reference system."""
    traj = reference_trajectory
    assert traj.v_over_c[0] == pytest.approx(0.025, rel=0.05)
    assert traj.v_over_c[-1] == pytest.approx(0.324, rel=0.05)
