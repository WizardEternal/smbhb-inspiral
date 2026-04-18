"""Regenerate all figures in the outputs/ directory.

Thin CLI wrapper around the package's public plotting functions.
Running this script reproduces every committed figure with no arguments.

Usage
-----
    PYTHONPATH=src python -m smbhb_inspiral.scripts.regenerate_all_figures

Outputs (written to ``outputs/``)
----------------------------------
* ``money_plot.png``       — dark-theme h_c(f) flagship figure
* ``money_plot_light.png`` — light-theme variant
"""

from __future__ import annotations

import pathlib
import sys


def main() -> None:
    """Regenerate every figure in outputs/ from the package's plotting API."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for headless / CI runs

    from smbhb_inspiral.plotting import make_money_plot

    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[str, dict]] = [
        ("money_plot.png",       {"theme": "dark"}),
        ("money_plot_light.png", {"theme": "light"}),
    ]

    for filename, kwargs in figures:
        out_path = output_dir / filename
        print(f"Generating {out_path} ...", flush=True)
        fig = make_money_plot(output_path=out_path, **kwargs)
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Saved -> {out_path}")

    print(f"\nAll figures written to {output_dir}/")


if __name__ == "__main__":
    main()
