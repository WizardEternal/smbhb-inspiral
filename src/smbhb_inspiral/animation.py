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
- Default masses are m1=7e6, m2=3e6 M☉, f₀=3 nHz, d_L=500 Mpc.  These were
  chosen so the chirp sweeps through *both* the PTA and LISA bands in one
  animation (f_ISCO ≈ 4.4e-4 Hz is squarely in the LISA band).  Pass the
  L6 reference values (m1=5e8, m2=2e8) if you want a PTA-only inspiral.

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
    m1_msun: float = 7e6,
    m2_msun: float = 3e6,
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
        Primary mass [M_sun]. Default 7×10⁶ (chosen so the chirp sweeps
        through both PTA and LISA bands in a single animation).
    m2_msun : float
        Secondary mass [M_sun]. Default 3×10⁶.
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

    # ------------------------------------------------------------------
    # Frame schedule: log-spaced in f_GW so each frame advances the chirp
    # by an equal fractional step in frequency.  Log-spacing in time
    # concentrates frames near merger by a factor that depends on the chirp
    # timescale (spanning 5+ orders of magnitude in f in the last 40% of
    # frames in practice), which makes the visible chirp look unevenly
    # paced. Log-f spacing gives a uniformly-perceived chirp rate.
    # ------------------------------------------------------------------
    t_start = float(traj.t[0])
    t_end = float(traj.t[-1])
    f_start = float(traj.f_gw[0])
    f_end = float(traj.f_gw[-1])
    log_f_frames = np.linspace(np.log(f_start), np.log(f_end), n_frames)
    f_gw_frames_requested = np.exp(log_f_frames)
    # Map back to times by inverting f_gw(t), monotone increasing
    frame_times = np.interp(f_gw_frames_requested, traj.f_gw, traj.t)

    # Interpolate physical state onto frame times.  Note: the trajectory
    # grid is log-spaced in TIME, so its final step covers a huge range in
    # f and a simultaneously — linearly interpolating traj.a in that last
    # segment gives values wildly inconsistent with Kepler.  Kepler-derived
    # a from f is used below where needed.
    f_gw_frames = f_gw_frames_requested
    v_over_c_frames = np.interp(frame_times, traj.t, traj.v_over_c)
    h_c_frames = np.interp(frame_times, traj.t, hc_full)
    t_norm_frames = frame_times / t_end  # for the f_GW(t) dot

    # h+ amplitude envelope at each frame, using the direct quadrupole form
    #     h(t) = 4 G μ ω² r² / (c⁴ D) · sin(2ωt)
    # with ω = π f_GW (orbital angular frequency, since f_GW = 2 f_orb).
    # Equivalent to the chirp-mass form under Kepler's law, but kept in this
    # shape for consistency with the St. Xavier's GR project (Akbari 2024).
    #
    # We derive the separation from f_GW via Kepler (rather than interpolating
    # traj.a) so the amplitude stays self-consistent with the instantaneous
    # frequency even where the log-spaced trajectory grid is sparse near
    # merger. Interpolating a and f independently can break their Kepler
    # relation across the last coalescence step, producing spurious amplitude
    # spikes and waveform artifacts.
    from .constants import G as _G, M_SUN, MPC, c as _c  # noqa: PLC0415
    mu_kg = (m1_msun * m2_msun) / (m1_msun + m2_msun) * M_SUN
    m_tot_kg = (m1_msun + m2_msun) * M_SUN
    m_c_kg = traj.chirp_mass_msun * M_SUN
    d_l_m = d_l_mpc * MPC
    omega_frames = np.pi * f_gw_frames  # orbital angular frequency [rad/s]
    a_kepler_frames = (_G * m_tot_kg / omega_frames ** 2) ** (1.0 / 3.0)
    h_amp_frames = (
        4.0 * _G * mu_kg * omega_frames ** 2 * a_kepler_frames ** 2
        / (_c ** 4 * d_l_m)
    )
    # Normalized separation for the orbital panel (Kepler-consistent)
    a_scale_frames = a_kepler_frames / a_kepler_frames[0]

    # ------------------------------------------------------------------
    # Stroboscopic orbital phase for the orbital-plane panel.
    # Physical phi(t) accumulates ~10^6 rad across the inspiral, so using
    # it directly aliases for >95% of frames. Instead, we render a visually
    # smooth rotation at a bounded rate (cycles_per_frame), decoupled from
    # physical time. The physical chirp is still honestly conveyed by the
    # f_GW(t) panel, the h_c track, and the v/c colormap.
    # ------------------------------------------------------------------
    cycles_per_frame = 1.0 / 12.0  # 12 frames per visual orbit
    phi_strobe_frames = 2.0 * np.pi * cycles_per_frame * np.arange(n_frames)

    # Pixel → inch conversion for matplotlib figsize. render_scale boosts the
    # saved PNG resolution without changing the figure layout or font sizes.
    dpi = 100
    render_scale = 1.6  # output at 1.6x logical resolution (~1280x960 default)
    fig_w_in = figsize[0] / dpi
    fig_h_in = figsize[1] / dpi

    # ------------------------------------------------------------------
    # Render frames
    # ------------------------------------------------------------------
    frames: list = []

    # Pre-compute stroboscopic BH positions for the orbital panel (constant
    # unit orbit, visually continuous rotation).
    pos1_strobe, pos2_strobe = _bh_positions(phi_strobe_frames, m1_msun, m2_msun)

    # Visual trail: last ~0.75 visual orbit at each frame, drawn as a dense
    # arc so the trail is a smooth curve rather than 5 discrete segments.
    N_trail_frames = max(1, int(round(0.75 / cycles_per_frame)))
    arc_resolution = 60  # points per full orbit of trail

    for frame_num in range(n_frames):
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

        current_voc = float(v_over_c_frames[frame_num])
        current_f_gw = float(f_gw_frames[frame_num])
        current_hc = float(h_c_frames[frame_num])
        current_h_amp = float(h_amp_frames[frame_num])
        current_phi_strobe = float(phi_strobe_frames[frame_num])
        bh_color = cmap(norm_voc(current_voc))

        # ------------------------------------------------------------------
        # Panel TL: Orbital plane (true inspiral — separation shrinks with a(t))
        # ------------------------------------------------------------------
        # Faint reference circle at initial separation so the viewer can see
        # the orbit shrink against a fixed-scale backdrop.
        ref_theta = np.linspace(0.0, 2.0 * np.pi, 180)
        ref_r1 = m2_msun / (m1_msun + m2_msun)
        ref_r2 = m1_msun / (m1_msun + m2_msun)
        ax_orb.plot(ref_r1 * np.cos(ref_theta), ref_r1 * np.sin(ref_theta),
                    color=colors["bh1"], alpha=0.12, lw=0.6, ls=":")
        ax_orb.plot(-ref_r2 * np.cos(ref_theta), -ref_r2 * np.sin(ref_theta),
                    color=colors["bh2"], alpha=0.12, lw=0.6, ls=":")

        # Inspiral trail: each past frame k rendered at its own a(t_k) so the
        # trail is a real shrinking spiral, not a circle.
        trail_frame_start = max(0, frame_num - N_trail_frames)
        trail_frames_range = np.arange(trail_frame_start, frame_num + 1)
        n_interp_per_frame = 6  # sub-samples between consecutive trail frames
        trail_t = np.linspace(
            float(trail_frames_range[0]), float(trail_frames_range[-1]),
            (len(trail_frames_range) - 1) * n_interp_per_frame + 1,
        )
        trail_phi = np.interp(trail_t, trail_frames_range.astype(float),
                              phi_strobe_frames[trail_frames_range])
        trail_a = np.interp(trail_t, trail_frames_range.astype(float),
                            a_scale_frames[trail_frames_range])
        pos1_arc, pos2_arc = _bh_positions(trail_phi, m1_msun, m2_msun)
        pos1_arc = pos1_arc * trail_a[:, None]
        pos2_arc = pos2_arc * trail_a[:, None]

        if len(trail_t) > 1:
            alphas = np.linspace(0.05, 0.55, len(trail_t))
            for k in range(len(trail_t) - 1):
                ax_orb.plot(
                    pos1_arc[k:k + 2, 0], pos1_arc[k:k + 2, 1],
                    color=colors["bh1"], alpha=float(alphas[k]), lw=0.9,
                )
                ax_orb.plot(
                    pos2_arc[k:k + 2, 0], pos2_arc[k:k + 2, 1],
                    color=colors["bh2"], alpha=float(alphas[k]), lw=0.9,
                )

        # Current BH positions, scaled by a(t)/a_0
        current_a_scale = float(a_scale_frames[frame_num])
        p1_now = pos1_strobe[frame_num] * current_a_scale
        p2_now = pos2_strobe[frame_num] * current_a_scale
        ax_orb.scatter(
            [p1_now[0]], [p1_now[1]],
            s=s1, color=colors["bh1"], edgecolors=bh_color,
            linewidths=1.5, zorder=5,
        )
        ax_orb.scatter(
            [p2_now[0]], [p2_now[1]],
            s=s2, color=colors["bh2"], edgecolors=bh_color,
            linewidths=1.5, zorder=5,
        )

        ax_orb.set_xlim(-0.8, 0.8)
        ax_orb.set_ylim(-0.8, 0.8)
        ax_orb.set_aspect("equal")
        ax_orb.set_title(
            f"Orbital plane  v/c={current_voc:.3f}  a/a₀={current_a_scale:.3f}",
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
        # Panel TR: h+ waveform — 5 GW cycles at the current (instantaneous)
        # f_GW, reconstructed from the direct quadrupole form
        #     h(t) = 4 G μ ω² r² / (c⁴ D) · sin(2ωt)
        # with a from Kepler at the current f.  The chirp (growing amplitude,
        # shrinking period) is conveyed by watching successive frames rather
        # than within one frame — this keeps the per-frame display a clean
        # sinusoid instead of a noisy chirp window.
        # ------------------------------------------------------------------
        n_gw_cycles = 5
        n_wave_samples = 400
        tau = np.linspace(0.0, n_gw_cycles / current_f_gw, n_wave_samples)
        h_wave = current_h_amp * np.sin(
            2.0 * np.pi * current_f_gw * tau + 2.0 * current_phi_strobe
        )
        t_norm = tau / tau[-1]
        ax_wave.plot(t_norm, h_wave, color=colors["wave"], lw=1.1)
        global_h_amp = float(np.max(h_amp_frames))
        ax_wave.set_ylim(-1.4 * global_h_amp, 1.4 * global_h_amp)
        ax_wave.axhline(0, color=colors["grid"], lw=0.5, ls="--")
        ax_wave.set_xlim(0, 1)
        ax_wave.set_xlabel(
            f"t  [{n_gw_cycles} GW cycles @ f={current_f_gw:.2e} Hz]",
            fontsize=6, color=colors["fg"],
        )
        ax_wave.set_ylabel(r"$h_+$", fontsize=7, color=colors["fg"])
        ax_wave.set_title(r"Waveform $h_+(t)$ — last 5 GW cycles",
                          fontsize=8, color=colors["fg"])

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

        # Full trajectory (faint) + portion up to now (by frame time, not index)
        t_all_norm = traj.t / t_end  # normalize to [0,1]
        ax_freq.plot(t_all_norm, traj.f_gw, color="#333355", lw=0.8, zorder=2)
        mask_past = traj.t <= frame_times[frame_num]
        ax_freq.plot(t_all_norm[mask_past], traj.f_gw[mask_past],
                     color=colors["fgw"], lw=1.2, zorder=3)

        # Moving dot at current frame time
        ax_freq.scatter(
            [t_norm_frames[frame_num]], [current_f_gw],
            s=40, color=colors["dot"], zorder=5,
        )

        # Axis spans both the PTA and LISA bands so the viewer can see where
        # this system sits relative to both detectors. For SMBHBs with total
        # mass ≳ 1e7 M☉, f_ISCO falls below the LISA floor — the empty LISA
        # band above the track makes that point visually.
        f_pad = 0.3
        f_lo = min(float(traj.f_gw[0]), 1e-9) * 10 ** (-f_pad)
        f_hi = 10 ** (-1 + f_pad)
        ax_freq.set_ylim(f_lo, f_hi)
        ax_freq.set_xlim(0, 1)
        ax_freq.set_xlabel("Time (normalized)", fontsize=6, color=colors["fg"])
        ax_freq.set_ylabel(r"$f_{\rm GW}$ [Hz]", fontsize=7, color=colors["fg"])
        ax_freq.set_title(r"Frequency $f_{\rm GW}(t)$", fontsize=8, color=colors["fg"])
        ax_freq.legend(fontsize=5, loc="upper left",
                       facecolor=colors["bg"], labelcolor=colors["fg"],
                       framealpha=0.5)

        # ------------------------------------------------------------------
        # Panel BR: h_c(f) with sensitivity curves + moving dot at current f
        # ------------------------------------------------------------------
        _static_hc_background(ax_hc, f_hc_full, hc_full, colors)

        ax_hc.scatter(
            [current_f_gw], [current_hc],
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
        fig.savefig(buf, format="png", dpi=dpi * render_scale,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    # ------------------------------------------------------------------
    # Save as GIF using Pillow. Per-frame durations: smoothly slow down
    # the final 30% of frames (up to 3x) so the late-inspiral chirp and
    # h_c track motion are readable, and hold on the last frame for 2.5 s
    # before the loop restarts.
    # ------------------------------------------------------------------
    interval_ms = int(1000 / fps)
    hold_ms = 2500
    n = len(frames)
    slow_start = int(0.7 * n)
    durations: list[int] = []
    for k in range(n):
        if k < slow_start:
            durations.append(interval_ms)
        else:
            # Linear ramp from 1x to 3x across the last 30%
            frac = (k - slow_start) / max(n - slow_start - 1, 1)
            durations.append(int(interval_ms * (1.0 + 2.0 * frac)))
    durations[-1] = hold_ms
    frames[0].save(
        outpath,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    return outpath.resolve()
