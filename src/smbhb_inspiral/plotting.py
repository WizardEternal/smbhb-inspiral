"""Flagship characteristic strain plot for SMBHB inspirals across GW detectors.

This module provides the "money plot" for the smbhb-inspiral package: a
log-log h_c(f) diagram that simultaneously shows the NANOGrav 15-yr and LISA
sensitivity curves alongside inspiral tracks for two reference SMBHB systems
— one in the PTA band and one in the LISA band — colored by orbital velocity
v/c.

The primary entry point is :func:`make_money_plot`.  A lower-level helper,
:func:`plot_strain_vs_detectors`, is provided for users who wish to build
custom figures.

References
----------
Robson, T., Cornish, N. J., & Liu, C. (2019).
    CQG, 36, 105011.
Agazie, G., et al. (NANOGrav Collaboration) (2023).
    ApJ Letters, 951, L8.
Sesana, A., Vecchio, A., & Colacino, C. N. (2008).
    MNRAS, 390, 192.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt

from .physics import integrate_inspiral, chirp_mass, f_isco, InspiralTrajectory
from .waveform import characteristic_strain_analytic
from .sensitivity import lisa_sensitivity_hc, nanograv_15yr_sensitivity_hc

__all__ = [
    "make_money_plot",
    "plot_strain_vs_detectors",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Frequency axis range for the full plot [Hz]
_F_MIN: float = 1e-10
_F_MAX: float = 1.0

# PTA and LISA band boundaries [Hz]
_PTA_FMIN: float = 1e-9
_PTA_FMAX: float = 1e-7
_LISA_FMIN: float = 1e-4
_LISA_FMAX: float = 1e-1


def _build_inspiral_track(
    m1: float,
    m2: float,
    f0: float,
    d_l_mpc: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Integrate an inspiral and return (f_gw, h_c, v_over_c).

    Parameters
    ----------
    m1 : float
        Primary mass [M_sun].
    m2 : float
        Secondary mass [M_sun].
    f0 : float
        Starting GW frequency [Hz].
    d_l_mpc : float
        Luminosity distance [Mpc].

    Returns
    -------
    f_gw : npt.NDArray[np.float64]
        GW frequency array [Hz].
    h_c : npt.NDArray[np.float64]
        Characteristic strain array (dimensionless).
    v_over_c : npt.NDArray[np.float64]
        Orbital velocity v/c array (dimensionless).
    """
    traj = integrate_inspiral(m1, m2, f0)
    f_gw = traj.f_gw
    h_c = characteristic_strain_analytic(
        f_hz=f_gw,
        chirp_mass_msun=traj.chirp_mass_msun,
        d_l_mpc=d_l_mpc,
    )
    return f_gw, h_c, traj.v_over_c


def _add_line_collection(
    ax: matplotlib.axes.Axes,
    f: npt.NDArray[np.float64],
    hc: npt.NDArray[np.float64],
    color_values: npt.NDArray[np.float64],
    norm: mcolors.Normalize,
    cmap: str,
    linewidth: float = 2.5,
) -> LineCollection:
    """Create a LineCollection on log-log axes, colored by *color_values*.

    The axis scales must already be set to 'log' before calling this helper.
    We work in data space; matplotlib's log transform is applied via
    ``transform=ax.transData``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (must have log scales set).
    f : npt.NDArray[np.float64]
        x-axis data (frequency) [Hz].
    hc : npt.NDArray[np.float64]
        y-axis data (characteristic strain).
    color_values : npt.NDArray[np.float64]
        Scalar values used to color each segment (same length as *f*).
    norm : mcolors.Normalize
        Normalization instance mapping color_values to [0, 1].
    cmap : str
        Matplotlib colormap name.
    linewidth : float
        Line width in points.

    Returns
    -------
    LineCollection
        The collection after it has been added to *ax*.
    """
    # Build (N-1, 2, 2) array of line segments in data coordinates.
    # Each segment connects consecutive (f, hc) pairs.
    # LineCollection on log axes: passing raw data works correctly because
    # ax.transData already incorporates the log10 transform.
    pts = np.column_stack([f, hc])           # shape (N, 2)
    segments = np.stack([pts[:-1], pts[1:]], axis=1)  # shape (N-1, 2, 2)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        transform=ax.transData,
        zorder=4,
    )
    lc.set_array(color_values[:-1])
    ax.add_collection(lc)
    return lc


def _theme_colors(theme: Literal["dark", "light"]) -> dict:
    """Return a dict of named colors for the chosen theme."""
    if theme == "dark":
        return {
            "bg": "#0a0a0a",
            "fg": "white",
            "grid": "#2a2a2a",
            "nanograv": "#888888",   # muted grey
            "lisa_curve": "#b0b0b0",  # muted silver/light grey
            "pta_band": "teal",
            "lisa_band": "gold",
            "annotation": "#aaaaaa",
        }
    else:  # light
        return {
            "bg": "white",
            "fg": "#111111",
            "grid": "#dddddd",
            "nanograv": "#555555",   # muted grey
            "lisa_curve": "#888888",  # muted silver/grey
            "pta_band": "teal",
            "lisa_band": "goldenrod",
            "annotation": "#555555",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_money_plot(
    output_path: str | Path | None = None,
    theme: Literal["dark", "light"] = "dark",
    figsize: tuple[float, float] = (14, 9),
    dpi: int = 150,
) -> Figure:
    """Generate the flagship h_c(f) plot with detector curves and inspiral tracks.

    This function runs two inspiral integrations internally (PTA and LISA
    reference systems), computes their characteristic strain tracks, and
    overlays them on NANOGrav 15-yr and LISA sensitivity curves.

    The two reference systems are:

    * **PTA reference**: m₁ = 5×10⁸ M☉, m₂ = 2×10⁸ M☉, D_L = 500 Mpc.
      Labeled "7×10⁸ M☉ PTA source (z≈0.1)".
    * **LISA reference**: m₁ = 2×10⁶ M☉, m₂ = 1×10⁶ M☉, D_L = 5000 Mpc.
      Labeled "3×10⁶ M☉ LISA source (z≈0.8)".

    Both tracks are colored by v/c using the ``inferno`` colormap (cool at
    low velocity / early inspiral, hot near ISCO).

    Parameters
    ----------
    output_path : path, optional
        If provided, save the figure to this path (PNG/PDF).  The file format
        is inferred from the extension.
    theme : {"dark", "light"}
        Color theme.  ``"dark"`` (default) uses a near-black background with
        white labels, suitable for README embedding.  ``"light"`` uses a
        white background with dark labels, suitable for papers and talks.
    figsize : tuple of float
        Figure size in inches ``(width, height)``.  Default is ``(14, 9)``.
    dpi : int
        Resolution in dots per inch used when saving to PNG.  Default is
        ``150``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The completed figure object.  The caller may further customize it or
        display it with ``plt.show()``.

    Notes
    -----
    The inspiral integrations start from the low-frequency end of each
    detector band and run to the ISCO.  Starting frequencies are chosen so
    that the tracks span the relevant detector band:

    * PTA track starts at ``f₀ = 2e-10`` Hz (below the NANOGrav band).
    * LISA track starts at ``f₀ = 5e-5`` Hz (within the LISA band).

    The v/c colormap normalization is shared across both tracks, with
    ``vmin = min(v/c)`` of the PTA track and ``vmax = max(v/c)`` of the
    LISA track (which ends closer to ISCO).

    Examples
    --------
    >>> from smbhb_inspiral.plotting import make_money_plot
    >>> fig = make_money_plot(theme="dark", output_path="money_plot.png")
    """
    colors = _theme_colors(theme)

    # ------------------------------------------------------------------
    # Reference systems
    # ------------------------------------------------------------------

    # PTA reference: 7e8 M_sun total, D_L = 500 Mpc (z ~ 0.1)
    M1_PTA, M2_PTA, DL_PTA = 5e8, 2e8, 500.0
    F0_PTA = 5e-10   # Start just below the NANOGrav band

    # LISA reference: 3e6 M_sun total (q=0.5), D_L = 5000 Mpc (z ~ 0.8)
    M1_LISA, M2_LISA, DL_LISA = 2e6, 1e6, 5000.0
    F0_LISA = 1e-5   # Start at the low edge of the LISA band

    f_pta, hc_pta, voc_pta = _build_inspiral_track(M1_PTA, M2_PTA, F0_PTA, DL_PTA)
    f_lisa, hc_lisa, voc_lisa = _build_inspiral_track(M1_LISA, M2_LISA, F0_LISA, DL_LISA)

    # Shared colormap normalization across both tracks
    voc_global_min = min(float(voc_pta.min()), float(voc_lisa.min()))
    voc_global_max = max(float(voc_pta.max()), float(voc_lisa.max()))
    norm = mcolors.Normalize(vmin=voc_global_min, vmax=voc_global_max)
    cmap_name = "inferno"

    # ------------------------------------------------------------------
    # Sensitivity curves
    # ------------------------------------------------------------------

    # NANOGrav 15-yr: digitized points
    f_ng, hc_ng = nanograv_15yr_sensitivity_hc()

    # LISA: analytic model on a fine log-spaced grid
    f_lisa_sens = np.geomspace(1e-5, 1.0, num=2000)
    hc_lisa_sens = lisa_sensitivity_hc(f_lisa_sens)

    # ------------------------------------------------------------------
    # Figure and axes setup
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(_F_MIN, _F_MAX)

    # Determine a sensible y-range after we have the curves
    _all_hc = np.concatenate([hc_pta, hc_lisa, hc_ng, hc_lisa_sens])
    _finite = _all_hc[np.isfinite(_all_hc) & (_all_hc > 0)]
    ymin = 10 ** (np.floor(np.log10(_finite.min())) - 0.5)
    ymax = 10 ** (np.ceil(np.log10(_finite.max())) + 0.5)
    ax.set_ylim(ymin, ymax)

    # ------------------------------------------------------------------
    # Band shading (drawn first so curves sit on top)
    # ------------------------------------------------------------------
    ax.axvspan(
        _PTA_FMIN, _PTA_FMAX,
        alpha=0.15, color=colors["pta_band"],
        label="PTA band", zorder=1,
    )
    ax.axvspan(
        _LISA_FMIN, _LISA_FMAX,
        alpha=0.15, color=colors["lisa_band"],
        label="LISA band", zorder=1,
    )

    # ------------------------------------------------------------------
    # Sensitivity curves
    # ------------------------------------------------------------------
    ax.plot(
        f_ng, hc_ng,
        color=colors["nanograv"],
        linewidth=2.0,
        linestyle="-",
        label="NANOGrav 15-yr (Agazie+2023)",
        zorder=3,
    )
    # Also show individual digitized points as small markers
    ax.plot(
        f_ng, hc_ng,
        color=colors["nanograv"],
        marker="o",
        markersize=3,
        linestyle="none",
        zorder=3,
    )

    ax.plot(
        f_lisa_sens, hc_lisa_sens,
        color=colors["lisa_curve"],
        linewidth=2.0,
        linestyle="-",
        label="LISA (Robson+2019)",
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Inspiral tracks (LineCollection, colored by v/c)
    # ------------------------------------------------------------------
    lc_pta = _add_line_collection(
        ax, f_pta, hc_pta, voc_pta, norm, cmap_name, linewidth=3.5
    )
    lc_lisa = _add_line_collection(
        ax, f_lisa, hc_lisa, voc_lisa, norm, cmap_name, linewidth=3.5
    )

    # Endpoint (ISCO) markers: color taken from the final v/c value
    cmap = plt.get_cmap(cmap_name)

    def _endpoint_color(voc_arr: npt.NDArray[np.float64]) -> tuple:
        return cmap(norm(float(voc_arr[-1])))

    ax.plot(
        f_pta[-1], hc_pta[-1],
        marker="o",
        markersize=8,
        color=_endpoint_color(voc_pta),
        zorder=5,
    )
    ax.plot(
        f_lisa[-1], hc_lisa[-1],
        marker="o",
        markersize=8,
        color=_endpoint_color(voc_lisa),
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Proxy artists for inspiral track legend entries
    # ------------------------------------------------------------------
    # Use a short horizontal line in the midpoint inferno color as proxy.
    _mid_pta_color = cmap(norm(float(np.median(voc_pta))))
    _mid_lisa_color = cmap(norm(float(np.median(voc_lisa))))

    proxy_pta = mlines.Line2D(
        [], [],
        color=_mid_pta_color,
        linewidth=2.5,
        label=r"$7\times10^8\,M_\odot$ PTA source ($z\approx0.1$)",
    )
    proxy_lisa = mlines.Line2D(
        [], [],
        color=_mid_lisa_color,
        linewidth=2.5,
        label=r"$3\times10^6\,M_\odot$ LISA source ($z\approx0.8$)",
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    # Collect existing handles + labels from ax (band spans + curves)
    handles, labels = ax.get_legend_handles_labels()
    # Append the track proxies
    handles.extend([proxy_pta, proxy_lisa])
    labels.extend([proxy_pta.get_label(), proxy_lisa.get_label()])

    leg = ax.legend(
        handles, labels,
        loc="upper right",
        fontsize=9,
        framealpha=0.3,
        facecolor=colors["bg"],
        edgecolor=colors["fg"],
        labelcolor=colors["fg"],
    )
    for text in leg.get_texts():
        text.set_color(colors["fg"])

    # ------------------------------------------------------------------
    # Colorbar (v/c)
    # ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
        aspect=30,
    )
    cbar.set_label(
        r"$v/c$",
        color=colors["fg"],
        fontsize=11,
    )
    cbar.ax.yaxis.set_tick_params(color=colors["fg"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=colors["fg"])
    cbar.outline.set_edgecolor(colors["fg"])

    # ------------------------------------------------------------------
    # Axis labels, ticks, grid
    # ------------------------------------------------------------------
    ax.set_xlabel("Gravitational-wave frequency  [Hz]", color=colors["fg"], fontsize=12)
    ax.set_ylabel("Characteristic strain  $h_c$", color=colors["fg"], fontsize=12)
    ax.tick_params(colors=colors["fg"], which="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["fg"])

    ax.grid(True, which="major", color=colors["grid"], linestyle="-", linewidth=0.6, zorder=0)
    ax.grid(True, which="minor", color=colors["grid"], linestyle=":", linewidth=0.3, zorder=0)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title(
        "Supermassive black hole binary inspirals across GW detectors",
        color=colors["fg"],
        fontsize=13,
        pad=10,
    )

    # ------------------------------------------------------------------
    # "Frequency gap" annotation between PTA and LISA bands
    # ------------------------------------------------------------------
    # Place it in the gap between 1e-7 and 1e-4 Hz, at a mid-height.
    gap_f = np.sqrt(_PTA_FMAX * _LISA_FMIN)  # geometric midpoint ~ 1e-5.5 Hz
    gap_y = np.sqrt(ymin * ymax) * 2.0        # somewhat above the plot centre

    ax.text(
        gap_f, gap_y,
        "frequency\ngap",
        color=colors["annotation"],
        fontsize=8,
        fontstyle="italic",
        ha="center",
        va="center",
        zorder=6,
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    return fig


# ---------------------------------------------------------------------------
# Lower-level helper for custom figures
# ---------------------------------------------------------------------------


def plot_strain_vs_detectors(
    ax: matplotlib.axes.Axes,
    trajectories: list[InspiralTrajectory],
    d_l_mpc_list: list[float],
    labels: list[str],
    show_nanograv: bool = True,
    show_lisa: bool = True,
    cmap: str = "inferno",
) -> matplotlib.axes.Axes:
    """Plot inspiral tracks on provided axes with optional detector curves.

    This is the lower-level counterpart to :func:`make_money_plot` for users
    who need full control over the figure layout, theming, or additional
    annotations.

    The axes must have log scales set (``ax.set_xscale('log')`` and
    ``ax.set_yscale('log')``) before calling this function, otherwise the
    :class:`~matplotlib.collections.LineCollection` coordinates will be
    interpreted as linear.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes on which to draw.  Should already have log scales set.
    trajectories : list of InspiralTrajectory
        Pre-computed inspiral trajectories (one per source).  Each must
        expose ``f_gw``, ``v_over_c``, and ``chirp_mass_msun``.
    d_l_mpc_list : list of float
        Luminosity distances in Mpc, one per trajectory.  Must be the same
        length as *trajectories*.
    labels : list of str
        Legend labels, one per trajectory.  Must be the same length as
        *trajectories*.
    show_nanograv : bool, optional
        If ``True`` (default), overlay the NANOGrav 15-yr sensitivity curve.
    show_lisa : bool, optional
        If ``True`` (default), overlay the LISA sensitivity curve.
    cmap : str, optional
        Matplotlib colormap name used to color track segments by v/c.
        Default is ``"inferno"``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axes object, with all elements added.  Returned for
        method-chaining convenience.

    Raises
    ------
    ValueError
        If *trajectories*, *d_l_mpc_list*, and *labels* do not all have the
        same length, or if *trajectories* is empty.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from smbhb_inspiral.physics import integrate_inspiral
    >>> from smbhb_inspiral.plotting import plot_strain_vs_detectors
    >>>
    >>> fig, ax = plt.subplots()
    >>> ax.set_xscale("log")
    >>> ax.set_yscale("log")
    >>>
    >>> traj = integrate_inspiral(1e8, 5e7, f0=1e-9)
    >>> plot_strain_vs_detectors(ax, [traj], [1000.0], ["1.5e8 Msun @ 1 Gpc"])
    >>> plt.show()
    """
    if len(trajectories) != len(d_l_mpc_list) or len(trajectories) != len(labels):
        raise ValueError(
            "trajectories, d_l_mpc_list, and labels must all have the same length; "
            f"got {len(trajectories)}, {len(d_l_mpc_list)}, {len(labels)}."
        )
    if not trajectories:
        raise ValueError("trajectories must contain at least one entry.")

    # ------------------------------------------------------------------
    # Compute h_c arrays for all trajectories
    # ------------------------------------------------------------------
    hc_arrays: list[npt.NDArray[np.float64]] = []
    voc_arrays: list[npt.NDArray[np.float64]] = []
    for traj, d_l in zip(trajectories, d_l_mpc_list):
        hc = characteristic_strain_analytic(
            f_hz=traj.f_gw,
            chirp_mass_msun=traj.chirp_mass_msun,
            d_l_mpc=d_l,
        )
        hc_arrays.append(hc)
        voc_arrays.append(traj.v_over_c)

    # Shared colormap normalization across all tracks
    voc_all = np.concatenate(voc_arrays)
    norm = mcolors.Normalize(vmin=float(voc_all.min()), vmax=float(voc_all.max()))
    cmap_obj = plt.get_cmap(cmap)

    # ------------------------------------------------------------------
    # Sensitivity curves
    # ------------------------------------------------------------------
    if show_nanograv:
        f_ng, hc_ng = nanograv_15yr_sensitivity_hc()
        ax.plot(
            f_ng, hc_ng,
            color="#888888",
            linewidth=2.0,
            linestyle="-",
            label="NANOGrav 15-yr (Agazie+2023)",
            zorder=3,
        )

    if show_lisa:
        f_lisa_sens = np.geomspace(1e-5, 1.0, num=2000)
        hc_lisa_sens = lisa_sensitivity_hc(f_lisa_sens)
        ax.plot(
            f_lisa_sens, hc_lisa_sens,
            color="#b0b0b0",
            linewidth=2.0,
            linestyle="-",
            label="LISA (Robson+2019)",
            zorder=3,
        )

    # ------------------------------------------------------------------
    # Inspiral tracks
    # ------------------------------------------------------------------
    for traj, hc, voc, label in zip(trajectories, hc_arrays, voc_arrays, labels):
        lc = _add_line_collection(
            ax, traj.f_gw, hc, voc, norm, cmap, linewidth=2.5
        )

        # Endpoint marker
        ep_color = cmap_obj(norm(float(voc[-1])))
        ax.plot(
            traj.f_gw[-1], hc[-1],
            marker="o",
            markersize=7,
            color=ep_color,
            zorder=5,
        )

        # Proxy artist for legend
        mid_color = cmap_obj(norm(float(np.median(voc))))
        proxy = mlines.Line2D([], [], color=mid_color, linewidth=2.5, label=label)
        ax.add_artist(  # type: ignore[call-arg]
            proxy
        )
        # Collect via legend handles manually below

    # ------------------------------------------------------------------
    # Legend (rebuild to include proxy entries)
    # ------------------------------------------------------------------
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    # Build proxy handles for the tracks
    track_proxies = [
        mlines.Line2D(
            [], [],
            color=cmap_obj(norm(float(np.median(voc)))),
            linewidth=2.5,
            label=label,
        )
        for voc, label in zip(voc_arrays, labels)
    ]
    all_handles = existing_handles + track_proxies
    all_labels = existing_labels + labels

    ax.legend(all_handles, all_labels, loc="upper right", fontsize=9)

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(r"$v/c$", fontsize=11)

    return ax
