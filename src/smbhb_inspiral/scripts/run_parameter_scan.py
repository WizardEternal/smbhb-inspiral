"""CLI parameter scan for a single SMBHB inspiral.

Thin wrapper around :func:`smbhb_inspiral.physics.integrate_inspiral` and
:func:`smbhb_inspiral.waveform.characteristic_strain_analytic`.  Accepts
binary parameters on the command line and prints a formatted summary.

Usage
-----
    PYTHONPATH=src python -m smbhb_inspiral.scripts.run_parameter_scan \\
        --m1 5e8 --m2 2e8 --f0 3e-9 --d-l-mpc 500

All physics lives in the package; this script contains no science.
"""

from __future__ import annotations

import argparse

import numpy as np


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_parameter_scan",
        description=(
            "Integrate a SMBHB inspiral for arbitrary parameters and print "
            "a formatted summary of key quantities."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m1",
        type=float,
        default=5e8,
        metavar="M_SUN",
        help="Primary black hole mass [solar masses].",
    )
    p.add_argument(
        "--m2",
        type=float,
        default=2e8,
        metavar="M_SUN",
        help="Secondary black hole mass [solar masses].",
    )
    p.add_argument(
        "--f0",
        type=float,
        default=3e-9,
        metavar="HZ",
        help="Initial gravitational-wave frequency [Hz].",
    )
    p.add_argument(
        "--d-l-mpc",
        type=float,
        default=500.0,
        metavar="MPC",
        help="Luminosity distance [Mpc].",
    )
    p.add_argument(
        "--pn-order",
        type=int,
        default=1,
        choices=[0, 1],
        metavar="{0,1}",
        help="Post-Newtonian order for the frequency evolution ODE.",
    )
    p.add_argument(
        "--ref-freq",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "Reference frequency at which to evaluate h_c [Hz]. "
            "Defaults to the geometric mean of f0 and f_ISCO."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for parameter-scan inspiral summary."""
    parser = _build_parser()
    args = parser.parse_args()

    m1: float = args.m1
    m2: float = args.m2
    f0: float = args.f0
    d_l_mpc: float = args.d_l_mpc
    pn_order: int = args.pn_order

    from smbhb_inspiral import chirp_mass, f_isco, integrate_inspiral
    from smbhb_inspiral.constants import YR
    from smbhb_inspiral.waveform import characteristic_strain_analytic

    m_chirp: float = chirp_mass(m1, m2)
    m_total: float = m1 + m2
    f_isco_val: float = f_isco(m_total)

    ref_freq: float = (
        args.ref_freq
        if args.ref_freq is not None
        else float(np.sqrt(f0 * f_isco_val))
    )

    # --- integrate ---
    traj = integrate_inspiral(m1, m2, f0, pn_order=pn_order)

    f_initial: float = float(traj.f_gw[0])
    f_final: float = float(traj.f_gw[-1])
    v_initial: float = float(traj.v_over_c[0])
    v_final: float = float(traj.v_over_c[-1])
    t_merge_yr: float = float(traj.t[-1]) / YR

    # characteristic strain at reference frequency
    h_c_ref: float = float(
        characteristic_strain_analytic(
            f_hz=np.array([ref_freq]),
            chirp_mass_msun=m_chirp,
            d_l_mpc=d_l_mpc,
        )[0]
    )

    # --- print summary ---
    sep = "=" * 60
    print(sep)
    print("  SMBHB Parameter Scan Summary")
    print(sep)
    print(f"  m1             = {m1:.3e}  M_sun")
    print(f"  m2             = {m2:.3e}  M_sun")
    print(f"  M_total        = {m_total:.3e}  M_sun")
    print(f"  M_chirp        = {m_chirp:.4e}  M_sun")
    print(f"  D_L            = {d_l_mpc:.3g}  Mpc")
    print(f"  PN order       = {pn_order}")
    print(sep)
    print(f"  f0 (initial)   = {f_initial:.3e}  Hz")
    print(f"  f_ISCO         = {f_isco_val:.3e}  Hz")
    print(f"  f_final        = {f_final:.3e}  Hz")
    print(f"  v/c (initial)  = {v_initial:.4f}")
    print(f"  v/c (final)    = {v_final:.4f}")
    print(f"  t_merge        = {t_merge_yr:.4g}  yr")
    print(f"  h_c @ {ref_freq:.2e} Hz = {h_c_ref:.3e}")
    print(sep)


if __name__ == "__main__":
    main()
