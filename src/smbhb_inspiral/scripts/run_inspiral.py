"""CLI entry point for a single SMBHB inspiral simulation.

Runs the post-Newtonian inspiral for a user-specified binary system, saves
a trajectory CSV, and (optionally) generates diagnostic plots.

Usage
-----
    python -m smbhb_inspiral.scripts.run_inspiral \\
        --m1 1e8 --m2 5e7 --f0 1e-8 --distance 500 --redshift 0.1

Outputs (in ``--output-dir``, default ``outputs/``)
----------------------------------------------------
* ``trajectory_<m1>_<m2>.csv``  — time-series of orbital evolution
* ``waveform.png``              — h+ vs time (last ~50 GW cycles)
* ``freq_evolution.png``        — f_gw vs time with PTA / LISA band shading
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from smbhb_inspiral import (
    chirp_mass,
    f_isco,
    integrate_inspiral,
)
from smbhb_inspiral.constants import G, M_SUN, MPC, PC, YR
from smbhb_inspiral.em_detectability import classify_system
from smbhb_inspiral.waveform import characteristic_strain_analytic


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _symmetric_mass_ratio(m1: float, m2: float) -> float:
    """Return eta = m1*m2 / (m1+m2)^2."""
    m_tot = m1 + m2
    return (m1 * m2) / (m_tot * m_tot)


def _mass_ratio(m1: float, m2: float) -> float:
    """Return q = m2/m1 where m1 >= m2 (q <= 1)."""
    return min(m1, m2) / max(m1, m2)


def _h_plus_from_trajectory(
    traj,
    d_l_mpc: float,
) -> np.ndarray:
    """Compute h+(t) from an InspiralTrajectory without using waveform.strain_plus.

    Uses the leading-order quadrupole formula (face-on, iota=0):

        h+(t) = A(t) * cos(phi(t))

    where A(t) = (4/D_L) * (G Mc/c^2)^{5/3} * (pi f_GW / c)^{2/3}.
    """
    from smbhb_inspiral.constants import c

    d_l_m: float = d_l_mpc * MPC
    m_c_kg: float = traj.chirp_mass_msun * M_SUN

    gm_over_c2: float = (G * m_c_kg) / c**2
    pi_f_over_c: np.ndarray = (np.pi * traj.f_gw) / c

    amp: np.ndarray = (
        (4.0 / d_l_m)
        * gm_over_c2 ** (5.0 / 3.0)
        * pi_f_over_c ** (2.0 / 3.0)
    )

    # face-on inclination factor: (1 + cos^2(0))/2 = 1.0
    return amp * np.cos(traj.phi)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_inspiral",
        description=(
            "Integrate a single SMBHB post-Newtonian inspiral and save "
            "trajectory data and diagnostic plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m1",
        type=float,
        required=True,
        metavar="M_SUN",
        help="Primary black hole mass [solar masses].",
    )
    p.add_argument(
        "--m2",
        type=float,
        required=True,
        metavar="M_SUN",
        help="Secondary black hole mass [solar masses].",
    )
    p.add_argument(
        "--f0",
        type=float,
        required=True,
        metavar="HZ",
        help="Initial gravitational-wave frequency [Hz].",
    )
    p.add_argument(
        "--distance",
        type=float,
        required=True,
        metavar="MPC",
        help="Luminosity distance to the source [Mpc].",
    )
    p.add_argument(
        "--redshift",
        type=float,
        default=0.0,
        metavar="Z",
        help=(
            "Source redshift (used for EM detectability classification). "
            "Set > 0 to enable the EM summary."
        ),
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
        "--output-dir",
        type=str,
        default="outputs",
        metavar="DIR",
        help="Directory in which to save output files.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help=(
            "Skip plot generation (useful for batch/headless runs). "
            "Trajectory CSV is always saved."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for single-system SMBHB inspiral."""
    parser = _build_parser()
    args = parser.parse_args()

    m1: float = args.m1
    m2: float = args.m2
    f0: float = args.f0
    d_l_mpc: float = args.distance
    z: float = args.redshift
    pn_order: int = args.pn_order
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    no_plots: bool = args.no_plots

    # --- derived system parameters ---
    m_total: float = m1 + m2
    m_chirp: float = chirp_mass(m1, m2)
    q: float = _mass_ratio(m1, m2)
    eta: float = _symmetric_mass_ratio(m1, m2)
    f_isco_val: float = f_isco(m_total)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  SMBHB Inspiral Simulation")
    print("=" * 60)
    print(f"  Primary mass           m1      = {m1:.3e}  M_sun")
    print(f"  Secondary mass         m2      = {m2:.3e}  M_sun")
    print(f"  Total mass             M_tot   = {m_total:.3e}  M_sun")
    print(f"  Chirp mass             M_chirp = {m_chirp:.3e}  M_sun")
    print(f"  Mass ratio             q       = {q:.4f}")
    print(f"  Symmetric mass ratio   eta     = {eta:.4f}")
    print(f"  Initial GW frequency   f0      = {f0:.3e}  Hz")
    print(f"  ISCO frequency         f_ISCO  = {f_isco_val:.3e}  Hz")
    print(f"  Luminosity distance    D_L     = {d_l_mpc:.3g}  Mpc")
    print(f"  PN order                       = {pn_order}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Integrate inspiral
    # -----------------------------------------------------------------------
    print("\nIntegrating inspiral...", flush=True)
    traj = integrate_inspiral(m1, m2, f0, pn_order=pn_order)
    print("Integration complete.")

    # -----------------------------------------------------------------------
    # Trajectory summary
    # -----------------------------------------------------------------------
    duration_yr: float = (traj.t[-1] - traj.t[0]) / YR
    f_gw_final: float = float(traj.f_gw[-1])
    v_over_c_final: float = float(traj.v_over_c[-1])
    a_initial_pc: float = float(traj.a[0]) / PC
    a_final_pc: float = float(traj.a[-1]) / PC

    # -----------------------------------------------------------------------
    # Characteristic strain track
    # -----------------------------------------------------------------------
    h_c_arr: np.ndarray = characteristic_strain_analytic(
        f_hz=traj.f_gw,
        chirp_mass_msun=traj.chirp_mass_msun,
        d_l_mpc=d_l_mpc,
    )
    h_c_peak: float = float(np.max(h_c_arr))

    print("\n--- Trajectory Summary ---")
    print(f"  Duration              : {duration_yr:.4g}  yr")
    print(f"  Final GW frequency    : {f_gw_final:.4e}  Hz")
    print(f"  Final v/c             : {v_over_c_final:.4f}")
    print(f"  Initial separation    : {a_initial_pc:.4g}  pc")
    print(f"  Final separation      : {a_final_pc:.4g}  pc")
    print(f"  Peak h_c (SPA)        : {h_c_peak:.3e}")

    # -----------------------------------------------------------------------
    # Save trajectory CSV
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"trajectory_{m1:.0e}_{m2:.0e}.csv"
    csv_path = output_dir / csv_name

    header = "t_s,f_gw_hz,f_orb_hz,a_m,v_over_c,phi_rad"
    trajectory_data = np.column_stack([
        traj.t,
        traj.f_gw,
        traj.f_orb,
        traj.a,
        traj.v_over_c,
        traj.phi,
    ])
    np.savetxt(
        csv_path,
        trajectory_data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8e",
    )
    print(f"\nTrajectory saved  -> {csv_path}")

    # -----------------------------------------------------------------------
    # EM detectability
    # -----------------------------------------------------------------------
    if z > 0.0:
        em_result = classify_system(
            m_total_msun=m_total,
            f_gw_hz=float(traj.f_gw[0]),
            z=z,
        )
        print("\n--- EM Detectability (Lin, Charisi & Haiman 2026) ---")
        print(f"  Rest-frame orbital period  : {em_result.p_rest_days:.2f}  days")
        print(f"  Observer-frame period      : {em_result.p_obs_days:.2f}  days  (z={z})")
        print(
            f"  In Stripe 82 window        : {em_result.in_stripe82_window}"
            + (
                f"  (sinusoidal {em_result.recovery_sinusoidal_stripe82*100:.0f}% / "
                f"sawtooth {em_result.recovery_sawtooth_stripe82*100:.0f}%)"
                if em_result.in_stripe82_window
                else ""
            )
        )
        print(
            f"  In PTF window              : {em_result.in_ptf_window}"
            + (
                f"  (sinusoidal {em_result.recovery_sinusoidal_ptf*100:.0f}% / "
                f"sawtooth {em_result.recovery_sawtooth_ptf*100:.0f}%)"
                if em_result.in_ptf_window
                else ""
            )
        )
        print(
            f"  In LSST window             : {em_result.in_lsst_window}"
            + (
                f"  (sinusoidal {em_result.recovery_sinusoidal_lsst*100:.0f}% / "
                f"sawtooth {em_result.recovery_sawtooth_lsst*100:.0f}%)"
                if em_result.in_lsst_window
                else ""
            )
        )

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    if not no_plots:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for headless runs
        import matplotlib.pyplot as plt

        system_title = (
            f"m1={m1:.1e} M☉, m2={m2:.1e} M☉, "
            f"D_L={d_l_mpc:.0f} Mpc, {pn_order}PN"
        )

        # --- Plot 1: Waveform (last ~50 GW cycles of h+) ---
        h_plus: np.ndarray = _h_plus_from_trajectory(traj, d_l_mpc)

        # Identify the last ~50 GW cycles.
        # One GW cycle corresponds to Delta_phi = 2*pi.
        # The orbital phase phi is the orbital phase; the GW phase is 2*phi
        # (since Phi_GW = 2 * phi_orb).  50 GW cycles => Delta_phi_gw = 100*pi
        # => Delta_phi_orb = 50*pi.
        phi_end = float(traj.phi[-1])
        phi_target = phi_end - 50.0 * np.pi  # 50 orbital cycles

        mask: np.ndarray
        if phi_target > float(traj.phi[0]):
            mask = traj.phi >= phi_target
        else:
            mask = np.ones(len(traj.phi), dtype=bool)

        t_wave = traj.t[mask]
        h_wave = h_plus[mask]
        # Shift time so the plot starts at t=0 for readability
        t_wave_shifted = t_wave - t_wave[0]

        with plt.style.context("dark_background"):
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(t_wave_shifted, h_wave, color="deepskyblue", lw=0.8)
            ax1.set_xlabel("Time [s]", fontsize=12)
            ax1.set_ylabel(r"$h_+$  (dimensionless)", fontsize=12)
            ax1.set_title(f"Waveform h+(t) — last ~50 cycles\n{system_title}", fontsize=11)
            ax1.grid(True, alpha=0.25)
            fig1.tight_layout()
            waveform_path = output_dir / "waveform.png"
            fig1.savefig(waveform_path, dpi=150)
            plt.close(fig1)
        print(f"Waveform plot saved -> {waveform_path}")

        # --- Plot 2: Frequency evolution ---
        t_yr = traj.t / YR  # convert to years for readability

        with plt.style.context("dark_background"):
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # PTA band: 1e-9 to 1e-7 Hz  (teal)
            ax2.axhspan(1e-9, 1e-7, alpha=0.25, color="teal", label="PTA band")

            # LISA band: 1e-4 to 1 Hz  (gold)
            ax2.axhspan(1e-4, 1.0, alpha=0.20, color="gold", label="LISA band")

            ax2.plot(t_yr, traj.f_gw, color="coral", lw=1.5, label="$f_{\\rm GW}(t)$")

            ax2.set_yscale("log")
            ax2.set_xlabel("Time [yr]", fontsize=12)
            ax2.set_ylabel(r"$f_{\rm GW}$  [Hz]", fontsize=12)
            ax2.set_title(
                f"Frequency evolution\n{system_title}", fontsize=11
            )
            ax2.legend(fontsize=10, loc="upper left")
            ax2.grid(True, which="both", alpha=0.20)
            fig2.tight_layout()
            freq_path = output_dir / "freq_evolution.png"
            fig2.savefig(freq_path, dpi=150)
            plt.close(fig2)
        print(f"Frequency plot saved -> {freq_path}")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    print(f"\nDone. Outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
