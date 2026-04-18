# Changelog

## [0.1.0] - Unreleased
### Added
- Core physics module: Peters (1964) + 1PN inspiral integrator
- Waveform module: h+, h×, characteristic strain
- Sensitivity curves: NANOGrav 15-yr + LISA (Robson+2019)
- EM detectability module: Lin, Charisi & Haiman (2026) recovery fractions
- CLI: run_inspiral.py
- Money plot: h_c(f) with dual detector curves
- **Phase 1c**: gap plot embed in `README.md` (`outputs/gap_plot_funnel.png`)
  — canonical synthetic seed=42 figure sourced from `tng-smbhb-population`,
  embedded per SESSION_HANDOFF.md line 122 ("goes in all three READMEs")
- `scripts/regenerate_all_figures.py`: thin CLI wrapper that regenerates all
  figures in `outputs/` by calling `smbhb_inspiral.plotting.make_money_plot`
  for both dark and light themes; runnable as
  `PYTHONPATH=src python -m smbhb_inspiral.scripts.regenerate_all_figures`
- `scripts/run_parameter_scan.py`: CLI that accepts `--m1`, `--m2`, `--f0`,
  `--d-l-mpc`, `--pn-order`, `--ref-freq` and prints a formatted summary
  (initial/final f_GW, v/c, t_merge, h_c at a reference frequency); runnable as
  `PYTHONPATH=src python -m smbhb_inspiral.scripts.run_parameter_scan`
- **Phase 2c**: CI/tooling hardening. `pyproject.toml` pin bounds verified;
  `.github/workflows/ci.yml` matrix aligned to `["3.11", "3.12"]` to match
  `requires-python = ">=3.11"` (previously mis-matrixed to 3.10, which would
  have failed pip install). `.pre-commit-config.yaml` added (ruff + black +
  mypy hooks). See also sibling repo CI configs for consistency.
- **Phase 2a**: 4-panel inspiral animation (`src/smbhb_inspiral/animation.py`, `scripts/make_animation.py`).
  Uses Pillow writer (no FFmpeg dependency). Defaults to L6 reference system.
  Renders `outputs/inspiral_animation.gif` at ~2.49 MB (under the 15 MB Gate 4 budget).
  Embedded in README as a magnet, not a novelty claim per L5.
- `docs/equations.md`: reviewer-facing reference documenting every physics
  equation in the package — Peters 1964 orbital evolution, 1PN correction,
  ISCO formula, GW strain polarizations, SPA characteristic strain,
  LISA Robson+2019 noise model, NANOGrav digitized curve, chirp mass, and
  Lin+2026 EM recovery fractions — with LaTeX, paper citations, and
  file:line cross-references
