"""GW detector sensitivity curves for characteristic strain diagrams.

This module provides analytic and data-driven sensitivity curves for two
gravitational-wave detector families relevant to supermassive black-hole
binary (SMBHB) inspirals:

* **LISA** — the Laser Interferometer Space Antenna (ESA/NASA), sensitive to
  mHz-band GWs from massive binaries in the final years before merger.
* **NANOGrav 15-year PTA** — the North American Nanohertz Observatory for
  Gravitational Waves dataset, sensitive to nHz-band GWs from the most massive
  binaries at cosmological distances.

**Important caveats — approximate / pedagogical use only**

The LISA curve uses the analytic noise model of Robson, Cornish & Liu (2019)
and is suitable for sensitivity-curve overlays.  It does *not* include the
(time-variable) confusion noise from Galactic compact binaries, which fills in
the trough around 3 mHz.

The NANOGrav curve is a rough digitization of the power-law integrated (PI)
sensitivity curve from Agazie et al. (2023).  These numbers are suitable for
educational plots and quick feasibility checks.  For production PTA analysis
use ``hasasia`` (Hazboun, Romano & Smith 2019,
https://github.com/Hazboun6/hasasia).

References
----------
Robson, T., Cornish, N. J., & Liu, C. (2019).
    The construction and use of LISA sensitivity curves.
    *Classical and Quantum Gravity*, 36, 105011.
    https://doi.org/10.1088/1361-6382/ab1101

Agazie, G., et al. (NANOGrav Collaboration) (2023).
    The NANOGrav 15-year Data Set: Evidence for a Gravitational-Wave Background.
    *ApJ Letters*, 951, L8.
    https://doi.org/10.3847/2041-8213/acdac6

Hazboun, J. S., Romano, J. D., & Smith, T. L. (2019).
    Realistic sensitivity curves for pulsar timing arrays.
    *Phys. Rev. D*, 100, 104028.
    https://doi.org/10.1103/PhysRevD.100.104028
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .constants import c

__all__ = [
    "lisa_sensitivity_hc",
    "nanograv_15yr_sensitivity_hc",
    "nanograv_15yr_sensitivity_hc_interp",
]

# ---------------------------------------------------------------------------
# LISA noise model parameters  (Robson, Cornish & Liu 2019)
# ---------------------------------------------------------------------------

#: LISA arm length [m].
_LISA_L: float = 2.5e9

#: LISA transfer frequency f_* = c / (2 pi L) [Hz].
#: Marks the transition from the low-frequency (Michelson) to the
#: high-frequency (transfer-function suppressed) regime.
_LISA_F_STAR: float = c / (2.0 * np.pi * _LISA_L)  # ≈ 19.09 mHz

# ---------------------------------------------------------------------------
# NANOGrav data file location
# ---------------------------------------------------------------------------

# The CSV lives at  src/smbhb_inspiral/data/nanograv_15yr_sensitivity.csv
# We use importlib.resources so it works whether the package is installed as
# an editable install, a wheel, or a zip-imported egg.
_DATA_PKG = "smbhb_inspiral.data"
_NANOGRAV_CSV = "nanograv_15yr_sensitivity.csv"


def _nanograv_csv_path() -> Path:
    """Return the resolved Path to the NANOGrav CSV file.

    Uses ``importlib.resources`` (Python 3.9+ traversable API) to locate the
    file inside the installed package, falling back to a ``__file__``-relative
    path for editable installs on older importlib versions.

    Returns
    -------
    Path
        Absolute path to ``nanograv_15yr_sensitivity.csv``.

    Raises
    ------
    FileNotFoundError
        If the data file cannot be located.
    """
    try:
        # Python 3.9+ path — works for installed wheels and editable installs.
        ref = importlib.resources.files(_DATA_PKG).joinpath(_NANOGRAV_CSV)
        # ``as_file`` context manager materialises zip-embedded resources to
        # a temporary file; for normal on-disk installs it just returns the
        # path directly.  We use the two-step form so the caller gets a plain
        # Path without needing a context manager.
        return Path(str(ref))
    except (AttributeError, ModuleNotFoundError):
        # Fallback for Python < 3.9 or unusual packaging situations.
        fallback = Path(__file__).parent / "data" / _NANOGRAV_CSV
        if not fallback.exists():
            raise FileNotFoundError(
                f"NANOGrav sensitivity CSV not found at expected location: {fallback}"
            )
        return fallback


# ---------------------------------------------------------------------------
# LISA sensitivity curve
# ---------------------------------------------------------------------------


def lisa_sensitivity_hc(
    f_hz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """LISA characteristic strain sensitivity curve.

    Implements the analytic noise model of Robson, Cornish & Liu (2019),
    CQG 36, 105011, Equations (1)–(3).  The result is sky-averaged and
    includes the instrument transfer function, making it directly comparable
    to the characteristic strain of a GW source.

    The one-sided power spectral density is:

    .. math::

       S_n(f) = \\frac{10}{3 L^2}
                \\left[
                  P_{\\rm OMS}(f)
                  + 2\\left(1 + \\cos^2\\frac{f}{f_*}\\right)
                    \\frac{P_{\\rm acc}(f)}{(2\\pi f)^4}
                \\right]
                \\left(1 + \\frac{6}{10}\\left(\\frac{f}{f_*}\\right)^2\\right)

    with the characteristic strain :math:`h_c(f) = \\sqrt{f \\, S_n(f)}`.

    The optical metrology noise (OMS) power spectrum is:

    .. math::

       P_{\\rm OMS}(f) = (1.5 \\times 10^{-11}\\,{\\rm m})^2
                         \\left[1 + \\left(\\frac{2 \\times 10^{-3}\\,{\\rm Hz}}{f}\\right)^4
                         \\right]\\,{\\rm Hz}^{-1}

    The test-mass acceleration noise power spectrum is:

    .. math::

       P_{\\rm acc}(f) = (3 \\times 10^{-15}\\,{\\rm m\\,s}^{-2})^2
                         \\left[1 + \\left(\\frac{0.4 \\times 10^{-3}\\,{\\rm Hz}}{f}\\right)^2
                         \\right]
                         \\left[1 + \\left(\\frac{f}{8 \\times 10^{-3}\\,{\\rm Hz}}\\right)^4
                         \\right]\\,{\\rm Hz}^{-1}

    Parameters
    ----------
    f_hz : npt.NDArray[np.float64]
        Frequencies in Hz at which to evaluate the sensitivity.
        Physical range: approximately ``1e-5`` to ``1`` Hz.
        Evaluating outside this range is numerically valid but physically
        unreliable (the analytic fit is not calibrated there).

    Returns
    -------
    h_c : npt.NDArray[np.float64]
        Characteristic strain sensitivity :math:`h_c(f) = \\sqrt{f S_n(f)}`,
        dimensionless, same shape as *f_hz*.

    Notes
    -----
    This model does **not** include the foreground confusion noise from
    Galactic compact binaries, which fills in the trough at ~3 mHz during
    certain mission phases.  For the full noise budget including the Galactic
    foreground, see Robson et al. (2019) Eq. (14).

    The :math:`(1 + 6/10 \\cdot (f/f_*)^2)` factor is the sky-averaged
    transfer function for an equal-arm Michelson interferometer (Larson,
    Hiscock & Hellings 2000).

    References
    ----------
    Robson, T., Cornish, N. J., & Liu, C. (2019).
        CQG, 36, 105011.  Eqs. (1)–(3).
    Larson, S. L., Hiscock, W. A., & Hellings, R. W. (2000).
        Phys. Rev. D, 62, 062001.
    """
    f: npt.NDArray[np.float64] = np.asarray(f_hz, dtype=np.float64)

    # --- Optical Metrology System (OMS) noise ---
    # Position noise floor: 1.5 pm / sqrt(Hz) with a 1/f^2 rise at low f.
    # Units after squaring: m^2 Hz^{-1}
    p_oms_amplitude_sq: float = (1.5e-11) ** 2  # m^2 Hz^{-1}
    p_oms: npt.NDArray[np.float64] = p_oms_amplitude_sq * (
        1.0 + (2.0e-3 / f) ** 4
    )

    # --- Test-mass acceleration noise ---
    # Residual force noise: 3 fm s^{-2} / sqrt(Hz) with low-f and high-f bumps.
    # Units after squaring: m^2 s^{-4} Hz^{-1}
    p_acc_amplitude_sq: float = (3.0e-15) ** 2  # m^2 s^{-4} Hz^{-1}
    p_acc: npt.NDArray[np.float64] = p_acc_amplitude_sq * (
        1.0 + (0.4e-3 / f) ** 2
    ) * (
        1.0 + (f / 8.0e-3) ** 4
    )

    # Convert acceleration noise to displacement noise by dividing by (2 pi f)^4.
    # Units become m^2 Hz^{-1}, matching P_OMS.
    two_pi_f_4: npt.NDArray[np.float64] = (2.0 * np.pi * f) ** 4
    p_acc_disp: npt.NDArray[np.float64] = p_acc / two_pi_f_4

    # --- Transfer function factor ---
    # Sky-averaged Michelson response (Larson et al. 2000; Robson et al. Eq. 3).
    f_ratio: npt.NDArray[np.float64] = f / _LISA_F_STAR
    transfer_factor: npt.NDArray[np.float64] = 1.0 + (6.0 / 10.0) * f_ratio ** 2

    # --- Noise PSD S_n(f)  [Hz^{-1}, strain^2] ---
    # The (10 / 3 L^2) prefactor converts from optical path-length fluctuations
    # to dimensionless strain, sky-averaged over source direction and polarization.
    cos_f_ratio: npt.NDArray[np.float64] = np.cos(f / _LISA_F_STAR)
    inner_bracket: npt.NDArray[np.float64] = (
        p_oms + 2.0 * (1.0 + cos_f_ratio ** 2) * p_acc_disp
    )

    s_n: npt.NDArray[np.float64] = (
        (10.0 / (3.0 * _LISA_L ** 2))
        * inner_bracket
        * transfer_factor
    )

    # --- Characteristic strain h_c(f) = sqrt(f * S_n(f)) ---
    h_c: npt.NDArray[np.float64] = np.sqrt(f * s_n)

    return h_c


# ---------------------------------------------------------------------------
# NANOGrav 15-year sensitivity curve
# ---------------------------------------------------------------------------


def _load_nanograv_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load the NANOGrav digitized sensitivity data from the package CSV.

    Returns
    -------
    f_hz : npt.NDArray[np.float64]
        Frequencies in Hz, sorted ascending.
    h_c : npt.NDArray[np.float64]
        Characteristic strain sensitivity values, same shape as *f_hz*.

    Raises
    ------
    FileNotFoundError
        If the bundled CSV cannot be found.
    ValueError
        If the CSV is malformed (wrong columns or non-numeric data).
    """
    csv_path = _nanograv_csv_path()

    # Parse the CSV manually: strip comment lines (starting with '#') and the
    # column-header row, then parse the remaining numeric rows.  This avoids
    # ambiguities in numpy.genfromtxt's interaction between ``comments``,
    # ``skip_header``, and ``names`` across numpy versions.
    rows: list[list[float]] = []
    with open(csv_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip the text header row (contains non-numeric content).
            parts = line.split(",")
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                # Non-numeric row (e.g. column header) — skip it.
                continue

    if not rows:
        raise ValueError(f"No numeric data rows found in {csv_path}.")

    arr = np.array(rows, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"Expected 2-column CSV (frequency_hz, h_c) at {csv_path}; "
            f"got shape {arr.shape}."
        )

    f_hz: npt.NDArray[np.float64] = arr[:, 0]
    h_c: npt.NDArray[np.float64] = arr[:, 1]

    # Ensure ascending frequency order (defensive, CSV is already sorted).
    order = np.argsort(f_hz)
    return f_hz[order], h_c[order]


def nanograv_15yr_sensitivity_hc() -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
):
    """NANOGrav 15-year characteristic strain sensitivity curve.

    Returns the digitized (f_hz, h_c) sensitivity array loaded from the
    bundled data file ``data/nanograv_15yr_sensitivity.csv``.  The values
    are approximate representations of the power-law integrated (PI)
    sensitivity curve published in Agazie et al. (2023), Figure 1.

    Returns
    -------
    f_hz : npt.NDArray[np.float64]
        GW frequencies in Hz, spanning approximately
        ``1e-9`` to ``1e-7`` Hz (the NANOGrav 15-yr sensitive band).
    h_c : npt.NDArray[np.float64]
        Characteristic strain sensitivity :math:`\\sqrt{f S_h(f)}` at each
        frequency in *f_hz*.  Minimum (best sensitivity) is approximately
        ``4e-15`` near ``f ~ 1e-8`` Hz.

    Notes
    -----
    **Provenance:** These are approximate values digitized by hand from
    Agazie et al. (2023), Figure 1.  They capture the gross shape of the
    PI sensitivity curve but are not a precise reproduction.  See
    ``data/PROVENANCE.md`` in the repository for details.

    For production-quality PTA sensitivity curves (including realistic
    noise models for individual pulsars) use ``hasasia``
    (Hazboun, Romano & Smith 2019, https://github.com/Hazboun6/hasasia).

    The PI sensitivity curve shown here represents the minimum detectable
    power-law GWB amplitude integrated over the NANOGrav frequency band.
    It is *not* the same as the noise PSD of a single pulsar.

    References
    ----------
    Agazie, G., et al. (NANOGrav Collaboration) (2023).
        ApJ Letters, 951, L8.  Figure 1.
        https://doi.org/10.3847/2041-8213/acdac6

    Hazboun, J. S., Romano, J. D., & Smith, T. L. (2019).
        Phys. Rev. D, 100, 104028.
        https://doi.org/10.1103/PhysRevD.100.104028
    """
    return _load_nanograv_data()


def nanograv_15yr_sensitivity_hc_interp(
    f_hz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolated NANOGrav 15-yr sensitivity at arbitrary frequencies.

    Performs log-log linear interpolation on the digitized sensitivity
    curve returned by :func:`nanograv_15yr_sensitivity_hc`.  Frequencies
    outside the digitized range return ``np.inf`` (not detectable /
    outside calibration range).

    Parameters
    ----------
    f_hz : npt.NDArray[np.float64]
        Frequencies in Hz at which to evaluate the sensitivity.
        Extrapolation is not performed; values outside ``[1e-9, 1e-7]`` Hz
        return ``np.inf``.

    Returns
    -------
    h_c_interp : npt.NDArray[np.float64]
        Interpolated characteristic strain sensitivity, same shape as
        *f_hz*.  Returns ``np.inf`` for any frequency outside the
        digitized range.

    Notes
    -----
    Log-log interpolation is appropriate here because both frequency and
    sensitivity span several decades, and the PI curve is locally a power
    law on each segment.

    ``np.inf`` is returned (rather than raising an error) for out-of-range
    frequencies so that the result can be safely used in signal-to-noise
    ratio computations:  ``SNR^2 = integral( (h_c_source / h_c_noise)^2 df )``,
    where ``np.inf`` in the denominator correctly contributes zero to the
    integral.

    References
    ----------
    Agazie, G., et al. (NANOGrav Collaboration) (2023).
        ApJ Letters, 951, L8.
        https://doi.org/10.3847/2041-8213/acdac6
    """
    f_in: npt.NDArray[np.float64] = np.asarray(f_hz, dtype=np.float64)
    f_data, hc_data = _load_nanograv_data()

    # Work in log space for the interpolation.
    log_f_data: npt.NDArray[np.float64] = np.log10(f_data)
    log_hc_data: npt.NDArray[np.float64] = np.log10(hc_data)
    log_f_in: npt.NDArray[np.float64] = np.log10(f_in)

    # Identify in-range points.
    in_range: npt.NDArray[np.bool_] = (log_f_in >= log_f_data[0]) & (
        log_f_in <= log_f_data[-1]
    )

    h_c_interp: npt.NDArray[np.float64] = np.full_like(f_in, fill_value=np.inf)
    if np.any(in_range):
        log_hc_interp = np.interp(
            log_f_in[in_range],
            log_f_data,
            log_hc_data,
        )
        h_c_interp[in_range] = 10.0 ** log_hc_interp

    return h_c_interp
