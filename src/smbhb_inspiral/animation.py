"""Four-panel SMBHB inspiral animation for README embedding.

Panels (2×2 grid, fixed layout):
  TL — orbital plane: two BH markers with fading trail, color-coded by v/c
  TR — h+ waveform: scrolling 5-cycle window
  BL — f_GW(t): log-scale frequency evolution with PTA/LISA bands + f_ISCO
  BR — h_c(f): static sensitivity curves + moving dot tracing the chirp

Design decisions
----------------
- Uses Pillow writer only — no FFmpeg dependency (Windows-hostile, see L4).
- Full-frame fig.savefig() loop instead of FuncAnimation blit, avoiding the
  blit+imshow class of bugs noted in EXECUTION_PLAN §7.1.
- Orbit radius is *normalized* (unit circle) so the orbital shrinkage is
  conveyed by the companion panels, not by visual collapse in Panel TL.
- Reference system default is L6: m1=5e8, m2=2e8 M☉, f₀=3 nHz, d_L=500 Mpc.

References
----------
Peters 1964, Phys. Rev. 136, B1224.
Sesana, Vecchio & Colacino 2008, MNRAS 390, 192.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt

from .physics import integrate_inspiral, f_isco
from .sensitivity import lisa_sensitivity_hc, nanograv_15yr_sensitivity_hc
from .waveform import characteristic_strain_track, strain_plus

__all__ = ["make_inspiral_animation"]

# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

def _theme(name: Literal["dark", "light"]) -> dict[str, str]:
    if name == "dark":
        return {
            "bg": "#0a0a0a", "fg": "white", "grid": "#2a2a2a",
            "trail": "#555555", "bh1": "#ff6b35", "bh2": "#35b8ff",
            "wave": "#7ecfb0", "fgw": "coral",
            "pta_band": "teal", "lisa_band": "gold",
            "ng_curve": "#888888", "lisa_curve": "#b0b0b0",
            "isco": "#ffcc44", "dot": "white",
        }
    else:
        return {
            "bg": "white", "fg": "#111111", "grid": "#dddddd",
            "trail": "#bbbbbb", "bh1": "#d44000", "bh2": "#0066cc",
            "wave": "#2a7a5a", "fgw": "#cc4400",
            "pta_band": "teal", "lisa_band": "goldenrod",
            "ng_curve": "#555555", "lisa_curve": "#888888",
            "isco": "#cc8800", "dot": "#111111",
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bh_positions(
    phi: npt.NDArray[np.float64],
    m1: float,
    m2: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (x1, y1) and (x2, y2) on a normalized unit circle.

    The two BHs sit at opposite sides of the CoM on a unit orbit.
    m1 sits at distance m2/(m1+m2) from CoM; m2 at distance m1/(m1+m2).
    Normalized so the orbit always appears full-size regardless of inspiral.
    """
    r1 = m2 / (m1 + m2)  # distance of m1 from CoM, in units of separation
    r2 = m1 / (m1 + m2)  # distance of m2 from CoM
    x1 = r1 * np.cos(phi)
    y1 = r1 * np.sin(phi)
    x2 = -r2 * np.cos(phi)
    y2 = -r2 * np.sin(phi)
    return np.stack([x1, y1], axis=1), np.stack([x2, y2], axis=1)


# ---------------------------------------------------------------------------
# Static background builders (built once, drawn every frame)
# ---------------------------------------------------------------------------

def _static_hc_background(
    ax: matplotlib.axes.Axes,
    f_hc: npt.NDArray[np.float64],
    hc_track: npt.NDArray[np.float64],
    colors: dict[str, str],
) -> None:
    """Draw sensitivity curves + inspiral track (no dot) on the h_c panel."""
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Sensitivity curves
    f_ng, hc_ng = nanograv_15yr_sensitivity_hc()
    ax.plot(f_ng, hc_ng, color=colors["ng_curve"], lw=1.5, label="NANOGrav 15-yr")

    f_lisa_sens = np.geomspace(1e-5, 1.0, 800)
    hc_lisa_sens = lisa_sensitivity_hc(f_lisa_sens)
    ax.plot(f_lisa_sens, hc_lisa_sens, color=colors["lisa_curve"], lw=1.5, label="LISA")

    # Band shading
    ax.axvspan(1e-9, 1e-7, alpha=0.12, color=colors["pta_band"])
    ax.axvspan(1e-4, 1e-1, alpha=0.12, color=colors["lisa_band"])

    # Inspiral track (static, faint)
    ax.plot(f_hc, hc_track, color="#555577", lw=1.2, zorder=2)

    # Axis range
    all_hc = np.concatenate([hc_track, hc_ng, hc_lisa_sens])
    finite = all_hc[np.isfinite(all_hc) & (all_hc > 0)]
    ax.set_xlim(min(float(f_hc[0]) * 0.5, 1e-9), max(float(f_hc[-1]) * 2.0, 1e-3))
    ax.set_ylim(10 ** (np.floor(np.log10(finite.min())) - 0.3),
                10 ** (np.ceil(np.log10(finite.max())) + 0.3))

    ax.set_xlabel("f [Hz]", fontsize=7, color=colors["fg"])
    ax.set_ylabel(r"$h_c$", fontsize=7, color=colors["fg"])
    ax.set_title(r"$h_c(f)$ track", fontsize=8, color=colors["fg"])


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def make_inspiral_animation(
    m1_msun: float = 5e8,
    m2_msun: float = 2e8,
    f0_hz: float = 3e-9,
    d_l_mpc: float = 500.0,
    pn_order: int = 1,
    n_frames: int = 120,
    fps: int = 20,
    outpath: str | Path = "outputs/inspiral_animation.gif",
    theme: Literal["dark", "light"] = "dark",
    figsize: tuple[int, int] = (800, 600),
) -> Path:
    """Generate a 4-panel animated GIF of an SMBHB inspiral.

    Panels
    ------
    Top-left   : orbital plane, normalized unit orbit, BHs colored by v/c
    Top-right  : h+(t) waveform, scrolling 5-cycle window
    Bottom-left: f_GW(t) log-scale with PTA/LISA bands + f_ISCO dashed line
    Bottom-right: h_c(f) with NANOGrav + LISA sensitivity + moving dot

    Parameters
    ----------
    m1_msun : float
        Primary mass [M_sun]. Default 5×10⁸ (L6 reference system).
    m2_msun : float
        Secondary mass [M_sun]. Default 2×10⁸.
    f0_hz : float
        Initial GW frequency [Hz]. Default 3 nHz.
    d_l_mpc : float
        Luminosity distance [Mpc]. Default 500.
    pn_order : int
        Post-Newtonian order (0 or 1). Default 1.
    n_frames : int
        Number of animation frames. Default 120.
    fps : int
        Frames per second for the output GIF. Default 20.
    outpath : str or Path
        Output path for the GIF file. Default ``outputs/inspiral_animation.gif``.
    theme : {"dark", "light"}
        Color theme. Default ``"dark"``.
    figsize : tuple of int
        Figure size in pixels ``(width, height)``. Default ``(800, 600)``.

    Returns
    -------
    Path
        Resolved absolute path to the written GIF file.

    Notes
    -----
    Uses Pillow as the animation writer (no FFmpeg dependency).
    The orbit is rendered at normalized unit radius so the visual scale stays
    stable; orbital shrinkage is conveyed by the frequency and h_c panels.

    References
    ----------
    Peters, P. C. 1964, Phys. Rev. 136, B1224.
    Sesana, A., Vecchio, A. & Colacino, C. N. 2008, MNRAS, 390, 192.
    """
    matplotlib.use("Agg")

    colors = _theme(theme)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Integrate full inspiral trajectory once
    # ------------------------------------------------------------------
    traj = integrate_inspiral(m1_msun, m2_msun, f0_hz, pn_order=pn_order)

    # Frame indices: evenly spaced across the full trajectory
    n_traj = len(traj.t)
    frame_indices = np.linspace(0, n_traj - 1, n_frames, dtype=int)

    # Pre-compute full-trajectory h+ and h_c track once
    h_plus_full: npt.NDArray[np.float64] = strain_plus(traj, d_l_mpc)
    f_hc_full, hc_full = characteristic_strain_track(traj, d_l_mpc)

    f_isco_val: float = f_isco(m1_msun + m2_msun)

    # BH marker sizes proportional to mass (scaled for display)
    s1 = 120.0 * (m1_msun / (m1_msun + m2_msun)) ** (1.0 / 3.0)
    s2 = 120.0 * (m2_msun / (m1_msun + m2_msun)) ** (1.0 / 3.0)

    # Colormap for v/c
    cmap = plt.get_cmap("inferno")
    voc_min = float(traj.v_over_c.min())
    voc_max = float(traj.v_over_c.max())
    norm_voc = mcolors.Normalize(vmin=voc_min, vmax=voc_max)

    # Trail length: last N_trail frames worth of orbit steps
    N_trail = max(1, n_traj // 40)

    # Pre-compute BH positions for all trajectory points
    pos1_all, pos2_all = _bh_positions(traj.phi, m1_msun, m2_msun)

    # h+ waveform: 5-cycle window width in phase units (5 GW cycles = 10π orbital)
    window_phase = 10.0 * math.pi  # 5 GW cycles

    # Pixel → inch conversion for matplotlib figsize
    dpi = 100
    fig_w_in = figsize[0] / dpi
    fig_h_in = figsize[1] / dpi

    # ------------------------------------------------------------------
    # Render frames
    # ------------------------------------------------------------------
    frames: list = []

    for frame_num, idx in enumerate(frame_indices):
        fig, axes = plt.subplots(
            2, 2,
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
        )
        fig.patch.set_facecolor(colors["bg"])
        for ax in axes.flat:
            ax.set_facecolor(colors["bg"])
            ax.tick_params(colors=colors["fg"], labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(colors["grid"])

        ax_orb, ax_wave, ax_freq, ax_hc = (
            axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        )

        current_voc = float(traj.v_over_c[idx])
        bh_color = cmap(norm_voc(current_voc))

        # ------------------------------------------------------------------
        # Panel TL: Orbital plane
        # ------------------------------------------------------------------
        trail_start = max(0, idx - N_trail)
        trail_idx = np.arange(trail_start, idx + 1)

        if len(trail_idx) > 1:
            # Fading trail: alpha ramps from 0 → 0.5
            alphas = np.linspace(0.05, 0.5, len(trail_idx))
            for k in range(len(trail_idx) - 1):
                seg = trail_idx[k:k + 2]
                ax_orb.plot(
                    pos1_all[seg, 0], pos1_all[seg, 1],
                    color=colors["bh1"], alpha=float(alphas[k]), lw=0.8,
                )
                ax_orb.plot(
                    pos2_all[seg, 0], pos2_all[seg, 1],
                    color=colors["bh2"], alpha=float(alphas[k]), lw=0.8,
                )

        # Current BH positions
        ax_orb.scatter(
            [pos1_all[idx, 0]], [pos1_all[idx, 1]],
            s=s1, color=colors["bh1"], edgecolors=bh_color,
            linewidths=1.5, zorder=5,
        )
        ax_orb.scatter(
            [pos2_all[idx, 0]], [pos2_all[idx, 1]],
            s=s2, color=colors["bh2"], edgecolors=bh_color,
            linewidths=1.5, zorder=5,
        )

        ax_orb.set_xlim(-0.8, 0.8)
        ax_orb.set_ylim(-0.8, 0.8)
        ax_orb.set_aspect("equal")
        ax_orb.set_title(
            f"Orbital plane  v/c={current_voc:.3f}",
            fontsize=8, color=colors["fg"],
        )
        ax_orb.set_xticks([])
        ax_orb.set_yticks([])

        # Colorbar strip (v/c) on the orbital panel
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm_voc)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_orb, fraction=0.04, pad=0.02)
        cbar.set_label("v/c", fontsize=6, color=colors["fg"])
        cbar.ax.tick_params(labelsize=5, colors=colors["fg"])
        cbar.outline.set_edgecolor(colors["fg"])

        # ------------------------------------------------------------------
        # Panel TR: h+ waveform (scrolling ~5 GW cycles)
        # ------------------------------------------------------------------
        phi_now = float(traj.phi[idx])
        phi_win_start = phi_now - window_phase
        mask_wave = traj.phi >= max(phi_win_start, float(traj.phi[0]))
        mask_wave &= traj.phi <= phi_now

        if np.sum(mask_wave) > 2:
            t_win = traj.t[mask_wave]
            h_win = h_plus_full[mask_wave]
            # Normalize time to [0, 1] within the window for a stable x-axis
            t_norm = (t_win - t_win[0]) / max(t_win[-1] - t_win[0], 1.0)
            ax_wave.plot(t_norm, h_win, color=colors["wave"], lw=0.9)
            h_amp = float(np.abs(h_win).max())
            if h_amp > 0:
                ax_wave.set_ylim(-1.4 * h_amp, 1.4 * h_amp)

        ax_wave.axhline(0, color=colors["grid"], lw=0.5, ls="--")
        ax_wave.set_xlim(0, 1)
        ax_wave.set_xlabel("Phase window (norm.)", fontsize=6, color=colors["fg"])
        ax_wave.set_ylabel(r"$h_+$", fontsize=7, color=colors["fg"])
        ax_wave.set_title(r"Waveform $h_+(t)$ — last 5 cycles", fontsize=8,
                          color=colors["fg"])

        # ------------------------------------------------------------------
        # Panel BL: Frequency evolution
        # ------------------------------------------------------------------
        ax_freq.set_yscale("log")

        # Band shading
        ax_freq.axhspan(1e-9, 1e-7, alpha=0.15, color=colors["pta_band"],
                        label="PTA band")
        ax_freq.axhspan(1e-4, 1e-1, alpha=0.15, color=colors["lisa_band"],
                        label="LISA band")

        # f_ISCO dashed line
        ax_freq.axhline(f_isco_val, color=colors["isco"], lw=1.0, ls="--",
                        label=f"$f_{{\\rm ISCO}}$={f_isco_val:.1e} Hz")

        # Full trajectory (faint) + portion up to now
        t_all_norm = traj.t / traj.t[-1]  # normalize to [0,1]
        ax_freq.plot(t_all_norm, traj.f_gw, color="#333355", lw=0.8, zorder=2)
        ax_freq.plot(t_all_norm[:idx + 1], traj.f_gw[:idx + 1],
                     color=colors["fgw"], lw=1.2, zorder=3)

        # Moving dot
        ax_freq.scatter(
            [t_all_norm[idx]], [float(traj.f_gw[idx])],
            s=40, color=colors["dot"], zorder=5,
        )

        f_pad = 0.5
        ax_freq.set_ylim(
            10 ** (math.log10(float(traj.f_gw[0])) - f_pad),
            10 ** (math.log10(f_isco_val) + f_pad),
        )
        ax_freq.set_xlim(0, 1)
        ax_freq.set_xlabel("Time (normalized)", fontsize=6, color=colors["fg"])
        ax_freq.set_ylabel(r"$f_{\rm GW}$ [Hz]", fontsize=7, color=colors["fg"])
        ax_freq.set_title(r"Frequency $f_{\rm GW}(t)$", fontsize=8, color=colors["fg"])
        ax_freq.legend(fontsize=5, loc="upper left",
                       facecolor=colors["bg"], labelcolor=colors["fg"],
                       framealpha=0.5)

        # ------------------------------------------------------------------
        # Panel BR: h_c(f) with sensitivity curves + moving dot
        # ------------------------------------------------------------------
        _static_hc_background(ax_hc, f_hc_full, hc_full, colors)

        # Moving dot: current position on the h_c track
        ax_hc.scatter(
            [float(f_hc_full[idx])], [float(hc_full[idx])],
            s=45, color=colors["dot"], zorder=6,
        )

        # ------------------------------------------------------------------
        # Suptitle with frame counter
        # ------------------------------------------------------------------
        pct = 100.0 * frame_num / max(n_frames - 1, 1)
        fig.suptitle(
            f"SMBHB Inspiral  "
            f"m₁={m1_msun:.1e} + m₂={m2_msun:.1e} M☉  "
            f"D_L={d_l_mpc:.0f} Mpc  "
            f"({pct:.0f}%)",
            fontsize=8, color=colors["fg"], y=0.995,
        )

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))

        # Capture frame as PIL image
        fig.canvas.draw()
        import io
        from PIL import Image  # noqa: PLC0415

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    # ------------------------------------------------------------------
    # Save as GIF using Pillow
    # ------------------------------------------------------------------
    interval_ms = int(1000 / fps)
    frames[0].save(
        outpath,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=interval_ms,
        loop=0,
        optimize=True,
    )

    return outpath.resolve()
