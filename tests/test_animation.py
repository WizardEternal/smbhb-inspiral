"""Tests for the smbhb_inspiral.animation module.

Verifies that the animation renders to a non-empty GIF without raising
exceptions.  Visual correctness is regression-tested by eyeball.

Marked with pytest.mark.slow so it can be deselected with -m "not slow":
    pytest tests/ -m "not slow"
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_make_inspiral_animation_renders(tmp_path: Path) -> None:
    """Render a minimal 10-frame animation and assert the GIF exists and is non-empty.

    Uses a tiny 320×240 canvas to keep runtime under a few seconds.
    The default PTA+LISA sweep system is used (m1=7e6, m2=3e6 M☉, f0=3 nHz).
    """
    import matplotlib
    matplotlib.use("Agg")

    from smbhb_inspiral.animation import make_inspiral_animation

    out = make_inspiral_animation(
        n_frames=10,
        figsize=(320, 240),
        outpath=tmp_path / "test_animation.gif",
    )

    assert out.exists(), f"Expected GIF at {out} but file was not created."
    assert os.path.getsize(out) > 0, "GIF file was created but is empty."
    # Sanity-check: a 10-frame GIF at this size should be well under 1 MB
    size_kb = os.path.getsize(out) / 1024
    assert size_kb < 1024, f"GIF unexpectedly large: {size_kb:.1f} KB"
