"""CLI wrapper for the 4-panel SMBHB inspiral animation.

Usage
-----
    python -m smbhb_inspiral.scripts.make_animation
    python -m smbhb_inspiral.scripts.make_animation --n-frames 60 --theme light
    python -m smbhb_inspiral.scripts.make_animation --m1 5e8 --m2 2e8 --f0 3e-9 \\
        --distance 500 --outpath outputs/my_animation.gif

All parameters default to the L6 reference system (m1=5×10⁸, m2=2×10⁸ M☉,
f₀=3 nHz, d_L=500 Mpc) per EXECUTION_PLAN.md §7.1.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_animation",
        description="Render a 4-panel SMBHB inspiral animation as a GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--m1", type=float, default=5e8, metavar="M_SUN",
        help="Primary BH mass [solar masses].",
    )
    p.add_argument(
        "--m2", type=float, default=2e8, metavar="M_SUN",
        help="Secondary BH mass [solar masses].",
    )
    p.add_argument(
        "--f0", type=float, default=3e-9, metavar="HZ",
        help="Initial GW frequency [Hz].",
    )
    p.add_argument(
        "--distance", type=float, default=500.0, metavar="MPC",
        help="Luminosity distance [Mpc].",
    )
    p.add_argument(
        "--pn-order", type=int, default=1, choices=[0, 1],
        help="Post-Newtonian order.",
    )
    p.add_argument(
        "--n-frames", type=int, default=120, metavar="N",
        help="Number of animation frames.",
    )
    p.add_argument(
        "--fps", type=int, default=20, metavar="FPS",
        help="Frames per second.",
    )
    p.add_argument(
        "--outpath", type=str, default="outputs/inspiral_animation.gif",
        metavar="PATH",
        help="Output GIF path.",
    )
    p.add_argument(
        "--theme", default="dark", choices=["dark", "light"],
        help="Color theme.",
    )
    return p


def main() -> None:
    """Entry point for the make_animation CLI."""
    import matplotlib
    matplotlib.use("Agg")

    from smbhb_inspiral.animation import make_inspiral_animation

    parser = _build_parser()
    args = parser.parse_args()

    print(f"Rendering {args.n_frames}-frame animation -> {args.outpath}")
    print(
        f"  System: m1={args.m1:.2e} + m2={args.m2:.2e} Msun, "
        f"f0={args.f0:.2e} Hz, D_L={args.distance:.0f} Mpc, {args.pn_order}PN"
    )

    out = make_inspiral_animation(
        m1_msun=args.m1,
        m2_msun=args.m2,
        f0_hz=args.f0,
        d_l_mpc=args.distance,
        pn_order=args.pn_order,
        n_frames=args.n_frames,
        fps=args.fps,
        outpath=args.outpath,
        theme=args.theme,
    )

    size_bytes = os.path.getsize(out)
    size_mb = size_bytes / 1e6
    print(f"Done. Output: {out}")
    print(f"File size: {size_bytes:,} bytes ({size_mb:.2f} MB)")
    if size_mb < 15.0:
        print(f"Gate 4 passed: {size_mb:.2f} MB < 15 MB")
    else:
        print(f"WARNING: Gate 4 FAILED — {size_mb:.2f} MB >= 15 MB cap")


if __name__ == "__main__":
    main()
