"""GW strain waveforms and characteristic strain for SMBHB inspirals.

This module computes the gravitational-wave strain polarizations and the
characteristic strain spectral amplitude for supermassive black-hole binary
(SMBHB) systems undergoing a quasi-circular inspiral driven by GW-radiation
reaction alone (no environmental effects).

**Scope and limitations**

* Orbits are assumed to be *circular* throughout the inspiral.  Eccentricity
  corrections (Peters & Mathews harmonics) are not included.
* The default inclination is ``iota = 0`` (face-on geometry), which maximises
  the received amplitude.  Arbitrary inclinations are supported through the
  ``iota`` parameter.
* The strain formulae are the leading-order (Newtonian) quadrupole expressions
  valid in the post-Newtonian inspiral regime, not the full merger–ringdown
  waveform.
* Characteristic strain uses the stationary-phase approximation (SPA) for a
  circular inspiral, appropriate for comparing against PTA / LISA sensitivity
  curves in the frequency domain.

**References**

* Maggiore, M. (2007). *Gravitational Waves: Theory and Experiments*,
  Oxford University Press.  Equations (4.100)–(4.101) for polarization
  amplitudes; (4.125) for the chirp form.
* Sesana, A., Vecchio, A., & Colacino, C. N. (2008).  The stochastic
  gravitational-wave background from massive black hole binary systems:
  implications for observations with Pulsar Timing Arrays.
  *MNRAS*, 390, 192.  Equation (2) for characteristic strain *h*_c.
* Flanagan, E. E. & Hughes, S. A. (1998).  Measuring gravitational-wave
  amplitudes from inspiraling compact binaries.  *Phys. Rev. D*, 57, 4535.
  Appendix B for the stationary-phase characteristic strain.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .constants import G, c, MPC
from .physics import InspiralTrajectory

__all__ = [
    "strain_plus",
    "strain_cross",
    "characteristic_strain_track",
    "characteristic_strain_analytic",
]

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _chirp_mass_kg(trajectory: InspiralTrajectory) -> float:
    """Return the chirp mass in kg from a trajectory's stored chirp mass.

    Parameters
    ----------
    trajectory:
        An :class:`~smbhb_inspiral.physics.InspiralTrajectory` instance.
        Its ``chirp_mass_msun`` attribute gives the chirp mass in solar masses.

    Returns
    -------
    float
        Chirp mass in kg.
    """
    from .constants import M_SUN  # keep import local to avoid circular issues
    return trajectory.chirp_mass_msun * M_SUN  # [kg]


def _amplitude_prefactor(
    m_c_kg: float,
    f_gw_hz: npt.NDArray[np.float64],
    d_l_m: float,
) -> npt.NDArray[np.float64]:
    """Compute the common amplitude factor A(t) = (4/D_L)(G M_c/c^2)^{5/3}(pi f_GW/c)^{2/3}.

    This is the dimensionless pre-factor shared by both polarizations before
    the inclination-dependent and phase-dependent terms are applied.

    Parameters
    ----------
    m_c_kg:
        Chirp mass in kg.
    f_gw_hz:
        GW frequency array in Hz.
    d_l_m:
        Luminosity distance in metres.

    Returns
    -------
    npt.NDArray[np.float64]
        Dimensionless amplitude array, same shape as *f_gw_hz*.

    Notes
    -----
    The formula follows Maggiore (2007) eq. (4.100).  The explicit factor
    structure is::

        A(t) = (4 / D_L) * (G M_c / c^2)^{5/3} * (pi f_GW / c)^{2/3}

    All quantities in SI units; the result is dimensionless.
    """
    # (G M_c / c^2) — "gravitational radius" of the chirp mass  [m]
    gm_over_c2: float = (G * m_c_kg) / c**2

    # (pi f_GW / c)  [m^{-1}]  — spatial wave-number factor
    pi_f_over_c: npt.NDArray[np.float64] = (np.pi * f_gw_hz) / c

    return (
        (4.0 / d_l_m)
        * gm_over_c2 ** (5.0 / 3.0)
        * pi_f_over_c ** (2.0 / 3.0)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strain_plus(
    trajectory: InspiralTrajectory,
    d_l_mpc: float,
    iota: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Plus polarization of the gravitational-wave strain h_+(t).

    Leading-order quadrupole expression for a quasi-circular inspiral:

    .. math::

       h_+(t) = \\frac{4}{D_L}
                \\left(\\frac{G\\mathcal{M}_c}{c^2}\\right)^{5/3}
                \\left(\\frac{\\pi f_{\\rm GW}}{c}\\right)^{2/3}
                \\frac{1 + \\cos^2\\iota}{2}\\,\\cos\\Phi(t)

    where :math:`D_L` is the luminosity distance, :math:`\\mathcal{M}_c` is
    the chirp mass, :math:`f_{\\rm GW} = 2 f_{\\rm orb}` is the GW frequency,
    and :math:`\\Phi(t)` is the accumulated orbital phase.

    Parameters
    ----------
    trajectory:
        Pre-computed inspiral trajectory from
        :func:`~smbhb_inspiral.physics.InspiralTrajectory`.  Must expose
        ``chirp_mass_msun`` (solar masses), ``f_gw`` (Hz array), and ``phi``
        (orbital phase, radians array).
    d_l_mpc:
        Luminosity distance to the source in megaparsecs.
    iota:
        Inclination angle between the binary orbital angular momentum and the
        line of sight, in radians.  ``iota = 0`` corresponds to a face-on
        system (maximum amplitude).  Default is ``0.0``.

    Returns
    -------
    npt.NDArray[np.float64]
        Time-series array of the plus-polarization strain h_+(t), dimensionless,
        same length as ``trajectory.f_gw``.

    Notes
    -----
    * All unit conversions are performed explicitly in SI:

      - ``d_l_mpc`` → metres via :data:`~smbhb_inspiral.constants.MPC`.
      - Chirp mass from solar masses to kg via
        :data:`~smbhb_inspiral.constants.M_SUN`.

    * The GW frequency is the *orbital* frequency doubled:
      :math:`f_{\\rm GW} = 2 f_{\\rm orb}`.  The trajectory is assumed to
      store ``f_gw`` already in these terms.

    References
    ----------
    Maggiore (2007), *Gravitational Waves*, eq. (4.100).
    """
    # --- unit conversions ---
    d_l_m: float = d_l_mpc * MPC  # Mpc → m

    m_c_kg: float = _chirp_mass_kg(trajectory)

    # --- compute amplitude ---
    amp: npt.NDArray[np.float64] = _amplitude_prefactor(
        m_c_kg, trajectory.f_gw, d_l_m
    )

    # inclination factor for + polarization: (1 + cos^2(iota)) / 2
    inc_factor: float = (1.0 + np.cos(iota) ** 2) / 2.0

    # phase modulation: cos(Phi(t))
    return amp * inc_factor * np.cos(trajectory.phi)


def strain_cross(
    trajectory: InspiralTrajectory,
    d_l_mpc: float,
    iota: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Cross polarization of the gravitational-wave strain h_×(t).

    Leading-order quadrupole expression for a quasi-circular inspiral:

    .. math::

       h_\\times(t) = \\frac{4}{D_L}
                      \\left(\\frac{G\\mathcal{M}_c}{c^2}\\right)^{5/3}
                      \\left(\\frac{\\pi f_{\\rm GW}}{c}\\right)^{2/3}
                      \\cos\\iota\\,\\sin\\Phi(t)

    Parameters
    ----------
    trajectory:
        Pre-computed inspiral trajectory from
        :func:`~smbhb_inspiral.physics.InspiralTrajectory`.  Must expose
        ``chirp_mass_msun`` (solar masses), ``f_gw`` (Hz array), and ``phi``
        (orbital phase, radians array).
    d_l_mpc:
        Luminosity distance to the source in megaparsecs.
    iota:
        Inclination angle between the binary orbital angular momentum and the
        line of sight, in radians.  ``iota = 0`` corresponds to a face-on
        system (circularly polarized maximum).  Default is ``0.0``.

    Returns
    -------
    npt.NDArray[np.float64]
        Time-series array of the cross-polarization strain h_×(t),
        dimensionless, same length as ``trajectory.f_gw``.

    Notes
    -----
    * The inclination factor for × polarization is :math:`\\cos\\iota`,
      compared with :math:`(1+\\cos^2\\iota)/2` for the + polarization.
    * For ``iota = 0`` (face-on), both polarizations have equal amplitude
      and the wave is circularly polarized.
    * For ``iota = pi/2`` (edge-on), the cross polarization vanishes.

    References
    ----------
    Maggiore (2007), *Gravitational Waves*, eq. (4.101).
    """
    # --- unit conversions ---
    d_l_m: float = d_l_mpc * MPC  # Mpc → m

    m_c_kg: float = _chirp_mass_kg(trajectory)

    # --- compute amplitude ---
    amp: npt.NDArray[np.float64] = _amplitude_prefactor(
        m_c_kg, trajectory.f_gw, d_l_m
    )

    # inclination factor for × polarization: cos(iota)
    inc_factor: float = float(np.cos(iota))

    # phase modulation: sin(Phi(t))
    return amp * inc_factor * np.sin(trajectory.phi)


def characteristic_strain_track(
    trajectory: InspiralTrajectory,
    d_l_mpc: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Characteristic strain track along an inspiral in the frequency domain.

    Returns arrays of (f_GW, h_c) evaluated at each point of the trajectory,
    suitable for overlaying on detector noise power spectral density (PSD) or
    sensitivity curves such as those of LISA or PTAs.

    Uses the stationary-phase approximation (SPA) for a circular inspiral:

    .. math::

       h_c(f) = \\frac{1}{D_L}
                \\sqrt{\\frac{2}{3}}
                \\frac{(G\\mathcal{M}_c)^{5/6}}{\\pi^{2/3}\\, c^{3/2}}
                f^{-1/6}

    Parameters
    ----------
    trajectory:
        Pre-computed inspiral trajectory.  Must expose ``chirp_mass_msun``
        (solar masses) and ``f_gw`` (Hz array).
    d_l_mpc:
        Luminosity distance to the source in megaparsecs.

    Returns
    -------
    f_hz : npt.NDArray[np.float64]
        GW frequency array in Hz, copied directly from ``trajectory.f_gw``.
    h_c : npt.NDArray[np.float64]
        Characteristic strain amplitude (dimensionless) evaluated at each
        frequency in *f_hz*.

    Notes
    -----
    The characteristic strain is related to the one-sided power spectral
    density of the strain signal by :math:`h_c^2(f) = 4 f^2 |\\tilde{h}(f)|^2`
    (Flanagan & Hughes 1998).

    The SPA result for a circular inspiral gives the spectrum
    :math:`h_c \\propto f^{-1/6}`, which is a gentle red tilt — much flatter
    than the :math:`f^{-7/6}` scaling of :math:`|\\tilde{h}(f)|` itself.

    This function is *sky-, polarization-, and inclination-averaged*
    over all orientations, as appropriate for sensitivity-curve
    comparisons.  The :math:`\\sqrt{2/3}/\\pi^{2/3}` prefactor encodes
    this average for circular orbits in the SPA (Moore, Cole & Berry
    2014, Eq. 14; equivalent to Sesana et al. 2008, Eq. 2).

    References
    ----------
    Moore, C. J., Cole, R. H. & Berry, C. P. L. (2014).
        Class. Quantum Grav. 32, 015014.  Equation (14).
    Sesana, A., Vecchio, A., & Colacino, C. N. (2008).
        MNRAS, 390, 192.  Equation (2).
    Flanagan, E. E. & Hughes, S. A. (1998).
        Phys. Rev. D, 57, 4535.  Appendix B.
    """
    f_hz: npt.NDArray[np.float64] = trajectory.f_gw  # Hz

    h_c: npt.NDArray[np.float64] = characteristic_strain_analytic(
        f_hz=f_hz,
        chirp_mass_msun=trajectory.chirp_mass_msun,
        d_l_mpc=d_l_mpc,
    )

    return f_hz, h_c


def characteristic_strain_analytic(
    f_hz: npt.NDArray[np.float64],
    chirp_mass_msun: float,
    d_l_mpc: float,
) -> npt.NDArray[np.float64]:
    """Characteristic strain h_c(f) for a circular inspiral on an arbitrary frequency grid.

    Uses the stationary-phase approximation for a quasi-circular GW-driven
    inspiral:

    .. math::

       h_c(f) = \\frac{1}{D_L}
                \\sqrt{\\frac{2}{3}}
                \\frac{(G\\mathcal{M}_c)^{5/6}}{\\pi^{2/3}\\, c^{3/2}}
                f^{-1/6}

    This is the sky-, polarization-, and inclination-averaged
    characteristic strain for a quasi-circular inspiral in the SPA,
    matching Moore, Cole & Berry (2014) Eq. 14 and Sesana et al.
    (2008) Eq. 2.

    Parameters
    ----------
    f_hz:
        Frequency array in Hz at which to evaluate h_c.  Must be positive.
    chirp_mass_msun:
        Chirp mass :math:`\\mathcal{M}_c` in solar masses.
    d_l_mpc:
        Luminosity distance in megaparsecs.

    Returns
    -------
    npt.NDArray[np.float64]
        Dimensionless characteristic strain h_c evaluated at each frequency
        in *f_hz*.  Same shape as *f_hz*.

    Notes
    -----
    Unit decomposition (SI throughout):

    .. code-block:: text

        [G M_c]^{5/6}  →  [m^3 s^{-2}]^{5/6}  =  m^{5/2} s^{-5/3}
        [c]^{3/2}      →  [m s^{-1}]^{3/2}     =  m^{3/2} s^{-3/2}
        [pi f]^{-1/6}  →  [s^{-1}]^{-1/6}      =  s^{1/6}
        [D_L]          →  [m]

        combined: m^{5/2-3/2} s^{-5/3+3/2+1/6} m^{-1}
                = m^1          s^{0}             m^{-1}
                = dimensionless  ✓

    The :math:`\\sqrt{2/3}/\\pi^{2/3}` prefactor is the sky-,
    polarization-, and inclination-averaged SPA result for circular
    orbits, normalized so that the matched-filter SNR is recovered via
    :math:`\\mathrm{SNR}^2 = \\int h_c^2 / h_n^2 \\,\\mathrm{d}\\ln f`
    with :math:`h_n^2(f) = f\\, S_n(f)`.

    References
    ----------
    Moore, C. J., Cole, R. H. & Berry, C. P. L. (2014).
        Class. Quantum Grav. 32, 015014.  Equation (14).
    Sesana, A., Vecchio, A., & Colacino, C. N. (2008).
        MNRAS, 390, 192.  Equation (2).
    Flanagan, E. E. & Hughes, S. A. (1998).
        Phys. Rev. D, 57, 4535.
    """
    from .constants import M_SUN  # local import to keep module-level imports clean

    # --- unit conversions ---
    d_l_m: float = d_l_mpc * MPC          # Mpc → m
    m_c_kg: float = chirp_mass_msun * M_SUN  # M_sun → kg

    # --- build the SPA characteristic strain ---
    # Numerator: (G M_c)^{5/6}  [m^{5/2} s^{-5/3}  — see Notes]
    gm_c: float = G * m_c_kg              # [m^3 s^{-2}]
    gm_c_pow: float = gm_c ** (5.0 / 6.0)

    # Denominator: D_L * c^{3/2} * pi^{2/3}
    c_pow: float = c ** (3.0 / 2.0)
    pi_pow: float = np.pi ** (2.0 / 3.0)

    # Frequency-dependent factor: f^{-1/6}
    freq_factor: npt.NDArray[np.float64] = f_hz ** (-1.0 / 6.0)  # [s^{1/6}]

    # Sky + inclination + polarization average prefactor for circular orbits
    # (Moore, Cole & Berry 2014, Eq. 14; Sesana et al. 2008, Eq. 2).
    avg_factor: float = np.sqrt(2.0 / 3.0)

    h_c: npt.NDArray[np.float64] = (
        (1.0 / d_l_m)
        * avg_factor
        * (gm_c_pow / c_pow)
        / pi_pow
        * freq_factor
    )

    return h_c
