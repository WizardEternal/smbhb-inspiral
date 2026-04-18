"""smbhb-inspiral: Post-Newtonian gravitational wave inspiral simulator for
supermassive black hole binaries.

Public API
----------
Physics
~~~~~~~
InspiralTrajectory   -- Container for an integrated inspiral solution
chirp_mass           -- Compute the chirp mass from component masses
f_isco               -- Innermost stable circular orbit frequency
integrate_inspiral   -- Integrate the Peters (1964) + 1PN equations of motion

Waveform
~~~~~~~~
characteristic_strain_analytic  -- Analytic h_c(f) for a circular binary
characteristic_strain_track     -- h_c along a numerically integrated track
strain_cross                    -- Cross polarisation h×(t)
strain_plus                     -- Plus polarisation h+(t)
"""

from smbhb_inspiral.physics import (
    InspiralTrajectory,
    chirp_mass,
    f_isco,
    integrate_inspiral,
)
from smbhb_inspiral.waveform import (
    characteristic_strain_analytic,
    characteristic_strain_track,
    strain_cross,
    strain_plus,
)

__version__ = "0.1.0"

__all__ = [
    # physics
    "InspiralTrajectory",
    "chirp_mass",
    "f_isco",
    "integrate_inspiral",
    # waveform
    "characteristic_strain_analytic",
    "characteristic_strain_track",
    "strain_cross",
    "strain_plus",
    # metadata
    "__version__",
]
