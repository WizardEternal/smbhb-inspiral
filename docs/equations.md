# Physics Equations — smbhb-inspiral

Reviewer-facing reference for every equation the package implements.
Each section cites the primary source and the file:line where the equation lives in code.

---

## 1. Chirp Mass

The chirp mass $\mathcal{M}_c$ is the combination of component masses that governs
the leading-order (0PN) phase evolution.  It is manifestly symmetric under
$m_1 \leftrightarrow m_2$.

$$
\mathcal{M}_c = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}
$$

- **Reference:** Peters & Mathews 1963, Phys. Rev. 131, 435, Eq. (3.16).
- **Implemented in:** `src/smbhb_inspiral/physics.py`, function `chirp_mass` (line ~93).
- **Reference system:** $m_1 = 5 \times 10^8\,M_\odot$, $m_2 = 2 \times 10^8\,M_\odot$
  gives $\mathcal{M}_c \approx 2.698 \times 10^8\,M_\odot$.

The symmetric mass ratio $\eta$ (used in the 1PN correction and stored in
`InspiralTrajectory.eta`) is

$$
\eta = \frac{m_1 m_2}{(m_1 + m_2)^2}
$$

with $\eta \in (0, 1/4]$; $\eta = 1/4$ for equal masses.

---

## 2. ISCO Frequency

For a Schwarzschild (non-spinning) black hole the innermost stable circular orbit
(ISCO) sits at $r_\mathrm{ISCO} = 6\,GM_\mathrm{tot}/c^2$.  Kepler's third law and
the relation $f_\mathrm{GW} = 2 f_\mathrm{orb}$ give

$$
f_\mathrm{ISCO} = \frac{c^3}{6^{3/2}\,\pi\,G\,M_\mathrm{tot}}
$$

- **Reference:** Peters 1964, Phys. Rev. 136, B1224; Misner, Thorne & Wheeler 1973,
  *Gravitation*, §33.5.
- **Implemented in:** `src/smbhb_inspiral/physics.py`, function `f_isco` (line ~163).
- **Reference values:**
  - $M_\mathrm{tot} = 7 \times 10^8\,M_\odot \Rightarrow f_\mathrm{ISCO} \approx 6.28 \times 10^{-6}\,\mathrm{Hz}$
    (falls in the **PTA–LISA frequency gap**, not inside the LISA band).
  - $M_\mathrm{tot} = 3 \times 10^6\,M_\odot \Rightarrow f_\mathrm{ISCO} \approx 1.47 \times 10^{-3}\,\mathrm{Hz}$
    (inside the LISA band).

The ISCO frequency is the default integration termination condition in
`integrate_inspiral`.

---

## 3. Orbital Evolution — Peters (1964) 0PN Term

The leading-order (quadrupole, Newtonian) rate of change of the gravitational-wave
frequency for a quasi-circular binary is (Peters 1964, Eq. 5.14):

$$
\left.\frac{df}{dt}\right|_\mathrm{0PN}
= \frac{96}{5}\,\pi^{8/3}
  \left(\frac{G\mathcal{M}_c}{c^3}\right)^{5/3} f^{11/3}
$$

This is an autonomous ODE in $f_\mathrm{GW}$ that drives frequency monotonically
upward (the inspiral sweeps from $f_0$ to $f_\mathrm{ISCO}$).

The orbital phase is integrated simultaneously:

$$
\frac{d\phi}{dt} = \pi\,f_\mathrm{GW}
$$

because the dominant GW harmonic is at twice the orbital frequency
($f_\mathrm{GW} = 2 f_\mathrm{orb}$), so the orbital angular velocity is
$\Omega_\mathrm{orb} = 2\pi f_\mathrm{orb} = \pi f_\mathrm{GW}$.

- **Reference:** Peters 1964, Phys. Rev. 136, B1224, Eq. (5.14).
- **Implemented in:** `src/smbhb_inspiral/physics.py`, function `peters_rhs`,
  variable `coeff_0pn` and `df_dt` (lines ~298–300).

### Analytic time-to-merger (0PN)

From Kepler's third law the initial orbital separation is

$$
a_0 = \left(\frac{G M_\mathrm{tot}}{4\pi^2 f_\mathrm{orb,0}^2}\right)^{1/3}
     = \left(\frac{G M_\mathrm{tot}}{\pi^2 f_0^2}\right)^{1/3}
$$

where $f_\mathrm{orb,0} = f_0/2$.  The analytic 0PN merge time is

$$
t_\mathrm{merge} = \frac{5}{256}
    \frac{c^5\,a_0^4}{G^3\,m_1\,m_2\,(m_1+m_2)}
$$

- **Reference:** Peters 1964, Phys. Rev. 136, B1224, Eq. (5.14).
- **Implemented in:** `src/smbhb_inspiral/physics.py`, function
  `analytic_t_merge_circular` (line ~571).
- **Reference value:** $(5 \times 10^8, 2 \times 10^8)\,M_\odot$ at $f_0 = 3\,\mathrm{nHz}$
  gives $t_\mathrm{merge} \approx 3.07 \times 10^{14}\,\mathrm{s} \approx 9.7\,\mathrm{Myr}$.

---

## 4. 1PN Correction to $df/dt$

The 1PN correction is applied as a multiplicative factor to the 0PN result.
With the dimensionless PN expansion parameter

$$
x = \left(\frac{\pi G M_\mathrm{tot} f_\mathrm{GW}}{c^3}\right)^{2/3}
$$

the corrected frequency derivative is

$$
\frac{df}{dt}\bigg|_\mathrm{1PN}
= \left.\frac{df}{dt}\right|_\mathrm{0PN}
  \times \left[1 - \left(\frac{743}{336} + \frac{11}{4}\,\eta\right) x \right]
$$

**Note:** the 1PN coefficient uses the **total mass** $M_\mathrm{tot}$, not the
chirp mass.  The 1PN correction reduces $df/dt$ (slows the frequency chirp) for
the mass ratios relevant to SMBHBs.

At $f_0 = 3\,\mathrm{nHz}$ with the reference system, $x \approx 4 \times 10^{-4}$
(weak-field), so the correction is sub-percent.  At ISCO ($f \approx 6\,\mu\mathrm{Hz}$)
it reaches $\sim\!8\%$, which is why 1PN is the default — the reference system
reaches $v/c \approx 0.32$ at ISCO, where 0PN is visibly wrong.

- **Reference:** Blanchet 2014, Living Rev. Relativity **17**, 2 (B14), §7.1, Eq. (234),
  spin-zero limit.
- **Implemented in:** `src/smbhb_inspiral/physics.py`, function `peters_rhs`,
  block `if pn_order >= 1:` (lines ~305–316).

---

## 5. Derived Orbital Quantities

After each integration step `integrate_inspiral` computes:

**Orbital frequency** (GW harmonic is at $2 f_\mathrm{orb}$):
$$
f_\mathrm{orb} = \frac{f_\mathrm{GW}}{2}
$$

**Orbital separation** via Kepler's third law:
$$
a = \left(\frac{G M_\mathrm{tot}}{4\pi^2 f_\mathrm{orb}^2}\right)^{1/3}
$$

**Dimensionless orbital velocity:**
$$
\frac{v}{c} = \left(\frac{\pi G M_\mathrm{tot} f_\mathrm{GW}}{2\,c^3}\right)^{1/3}
$$

- **Implemented in:** `src/smbhb_inspiral/physics.py`, function `integrate_inspiral`,
  post-integration block (lines ~532–547).

---

## 6. Gravitational-Wave Strain Polarizations

For a face-on ($\iota = 0$) quasi-circular inspiral at luminosity distance $D_L$,
the two GW polarizations are:

$$
h_+(t) = \frac{4}{D_L}
         \left(\frac{G\mathcal{M}_c}{c^2}\right)^{5/3}
         \left(\frac{\pi f_\mathrm{GW}}{c}\right)^{2/3}
         \frac{1 + \cos^2\iota}{2}\,\cos\Phi(t)
$$

$$
h_\times(t) = \frac{4}{D_L}
              \left(\frac{G\mathcal{M}_c}{c^2}\right)^{5/3}
              \left(\frac{\pi f_\mathrm{GW}}{c}\right)^{2/3}
              \cos\iota\,\sin\Phi(t)
$$

where $\Phi(t)$ is the accumulated orbital phase and $\iota$ is the inclination
angle (angle between orbital angular momentum and line of sight).  At $\iota = 0$
(face-on) both polarizations have equal amplitude and the wave is circularly
polarized.  At $\iota = \pi/2$ (edge-on) $h_\times = 0$.

These are the leading-order (Newtonian quadrupole) expressions, valid in the
inspiral regime only — no merger or ringdown.

- **Reference:** Maggiore 2007, *Gravitational Waves: Theory and Experiments*,
  Oxford University Press, Eqs. (4.100)–(4.101).
- **Implemented in:** `src/smbhb_inspiral/waveform.py`, functions `strain_plus`
  and `strain_cross` (lines ~123 and ~196).

---

## 7. Characteristic Strain

The characteristic strain $h_c(f)$ is defined via the one-sided power spectral
density of the strain signal:

$$
h_c^2(f) = 4 f^2 |\tilde{h}(f)|^2
$$

Under the stationary-phase approximation (SPA) for a quasi-circular, GW-driven
inspiral, this evaluates to

$$
h_c(f) = \frac{1}{\pi D_L}
          \sqrt{\frac{2}{3}}
          \frac{(G\mathcal{M}_c)^{5/6}}{c^{3/2}}
          (\pi f)^{-1/6}
$$

The spectrum scales as $h_c \propto f^{-1/6}$, which is a gentle red tilt — much
flatter than $|\tilde{h}(f)| \propto f^{-7/6}$.

The $\sqrt{2/3}$ prefactor is the RMS sky and inclination average for circularly
polarized waves (two independent polarizations with equal power), appropriate for
comparing against PTA / LISA sensitivity curves.

**Scope:** the SPA formula assumes the inspiral spends many cycles near each
frequency, which holds throughout the PTA and LISA bands for SMBHB systems.  It
is **not** valid during merger or ringdown, which are not modeled by this package.

- **Reference:** Sesana, Vecchio & Colacino 2008, MNRAS 390, 192, Eq. (2);
  Flanagan & Hughes 1998, Phys. Rev. D 57, 4535, App. B.
- **Implemented in:** `src/smbhb_inspiral/waveform.py`, function
  `characteristic_strain_analytic` (line ~329); called by
  `characteristic_strain_track` (line ~261).

---

## 8. LISA Sensitivity Curve — Robson, Cornish & Liu 2019

The analytic LISA noise model of Robson, Cornish & Liu 2019 (CQG 36, 105011)
is used.  The one-sided power spectral density is

$$
S_n(f) = \frac{10}{3L^2}
\left[
  P_\mathrm{OMS}(f)
  + 2\left(1 + \cos^2\frac{f}{f_*}\right)
    \frac{P_\mathrm{acc}(f)}{(2\pi f)^4}
\right]
\left(1 + \frac{6}{10}\left(\frac{f}{f_*}\right)^2\right)
$$

where $L = 2.5 \times 10^9\,\mathrm{m}$ is the LISA arm length and
$f_* = c / (2\pi L) \approx 19.09\,\mathrm{mHz}$ is the transfer frequency.

**Optical metrology system (OMS) noise** (position noise, $1.5\,\mathrm{pm}/\sqrt{\mathrm{Hz}}$):

$$
P_\mathrm{OMS}(f) = (1.5 \times 10^{-11}\,\mathrm{m})^2
\left[1 + \left(\frac{2 \times 10^{-3}\,\mathrm{Hz}}{f}\right)^4\right]
\,\mathrm{Hz}^{-1}
$$

**Test-mass acceleration noise** ($3\,\mathrm{fm\,s^{-2}}/\sqrt{\mathrm{Hz}}$):

$$
P_\mathrm{acc}(f) = (3 \times 10^{-15}\,\mathrm{m\,s^{-2}})^2
\left[1 + \left(\frac{0.4 \times 10^{-3}\,\mathrm{Hz}}{f}\right)^2\right]
\left[1 + \left(\frac{f}{8 \times 10^{-3}\,\mathrm{Hz}}\right)^4\right]
\,\mathrm{Hz}^{-1}
$$

The $(1 + 6/10\,(f/f_*)^2)$ factor is the sky-averaged Michelson transfer function
(Larson, Hiscock & Hellings 2000, Phys. Rev. D 62, 062001).

The characteristic strain is then

$$
h_c^\mathrm{LISA}(f) = \sqrt{f\,S_n(f)}
$$

**Caveats:** this model does **not** include Galactic compact-binary confusion
noise, which fills in the $\sim\!3\,\mathrm{mHz}$ trough during certain mission
phases.  For full noise budget use Robson et al. (2019) Eq. (14).

- **Reference:** Robson, Cornish & Liu 2019, CQG 36, 105011, Eqs. (1)–(3).
- **Implemented in:** `src/smbhb_inspiral/sensitivity.py`, function
  `lisa_sensitivity_hc` (line ~123).

---

## 9. NANOGrav 15-Year Sensitivity Curve

The NANOGrav 15-yr sensitivity is loaded from a digitized CSV file bundled with
the package:

```
src/smbhb_inspiral/data/nanograv_15yr_sensitivity.csv
```

Columns: `frequency_hz`, `h_c`.

The values are a manual digitization of the power-law-integrated (PI) sensitivity
curve from Agazie et al. 2023, ApJ Letters 951, L8, Figure 1.  They cover
approximately $1\,\mathrm{nHz}$ to $0.1\,\mu\mathrm{Hz}$ (the 15-yr PTA band).

An interpolated version is available via `nanograv_15yr_sensitivity_hc_interp`,
which performs bilinear (log-log linear) interpolation:

$$
\log_{10} h_c(f) = \mathrm{interp}\!\left(\log_{10} f;\;\{f_i, h_{c,i}\}\right)
$$

Frequencies outside the digitized range return `np.inf`, so that SNR integrals
over the full band contribute zero weight outside the calibrated range.

- **Reference:** Agazie et al. (NANOGrav Collaboration) 2023, ApJ Lett. 951, L8.
- **Digitization provenance:** `data/PROVENANCE.md` in the repository.
- **Implemented in:** `src/smbhb_inspiral/sensitivity.py`, functions
  `nanograv_15yr_sensitivity_hc` (line ~308) and
  `nanograv_15yr_sensitivity_hc_interp` (line ~356).

---

## 10. EM Detectability — Lin, Charisi & Haiman 2026

EM detectability is assessed by checking whether the observer-frame orbital period
falls within a survey's sensitivity window, then applying the Lomb-Scargle (LS)
recovery fraction published in Lin, Charisi & Haiman 2026 (ApJ 997, 316,
DOI [10.3847/1538-4357/ae29a7](https://doi.org/10.3847/1538-4357/ae29a7)).

### Period transformations

**GW frequency to orbital period** (quasi-circular, $f_\mathrm{GW} = 2 f_\mathrm{orb}$):

$$
P_\mathrm{rest} = \frac{2}{f_\mathrm{GW}}
$$

**Observer-frame period** (cosmological time dilation):

$$
P_\mathrm{obs} = P_\mathrm{rest}\,(1 + z)
$$

**Orbital period from separation** (Kepler's third law):

$$
P_\mathrm{orb} = 2\pi\sqrt{\frac{a^3}{G M_\mathrm{tot}}}
$$

- **Implemented in:** `src/smbhb_inspiral/em_detectability.py`, functions
  `orbital_period_from_f_gw`, `observer_frame_period`,
  `orbital_period_from_separation` (lines ~230, ~262, ~196).

### Survey sensitivity windows

A system is considered detectable if $P_\mathrm{min} \le P_\mathrm{obs} \le P_\mathrm{max}$.
The window boundaries are set by the 3-cycle requirement
($P_\mathrm{max} \approx \mathrm{baseline}/3$) and cadence-limited Nyquist
sampling ($P_\mathrm{min}$):

| Survey   | $P_\mathrm{min}$ (days) | $P_\mathrm{max}$ (days) | Baseline (days) | Cadence tier |
|----------|------------------------|------------------------|-----------------|--------------|
| Stripe82 | 200                    | 1100                   | 3650            | PTF-like     |
| PTF      | 100                    | 600                    | 1825            | PTF-like     |
| LSST     | 100                    | 1200                   | 3650            | LSST-like    |

- **Implemented in:** `src/smbhb_inspiral/em_detectability.py`, dict `SURVEY_WINDOWS`
  (line ~90).

### Locked recovery fractions (Lin+2026 Table 1, §3.1)

These values are frozen constants; do not modify without updating the citation.

| Signal shape | PTF-like | Idealized | LSST-like |
|-------------|----------|-----------|-----------|
| Sinusoidal  | 45%      | 24%       | 23%       |
| Sawtooth    |  9%      |  1%       |  1%       |

Because Stripe82 and PTF share PTF-like cadence, they map to the same column.

**Key implication (direct author quote):**
> "Previous searches, including the one in M. Charisi et al. (2016), must have missed a significant fraction of periodic signals."
> — Lin, Charisi & Haiman 2026, ApJ 997, 316

The sawtooth fractions (9/1/1%) reflect the difficulty of recovering
non-sinusoidal signals — the dominant waveform expected from SMBHBs when the
modulation is driven by Doppler shifts or relativistic effects.

- **Reference:** Lin, Charisi & Haiman 2026, ApJ 997, 316, Table 1, §3.1.
- **Implemented in:** `src/smbhb_inspiral/em_detectability.py`, dict
  `RECOVERY_FRACTIONS` (line ~58); applied by `classify_system` (line ~426) and
  `classify_system_from_separation` (line ~479).

---

## 11. Integrator Details

The state vector is $\mathbf{y} = [f_\mathrm{GW},\, \phi]$.  The ODE is integrated
with `scipy.integrate.solve_ivp` using the DOP853 scheme (Dormand & Prince 1980,
J. Comput. Appl. Math. 6, 19) — an explicit 8th-order Runge-Kutta method with
adaptive step-size control.  Default tolerances: $\mathrm{rtol} = 10^{-10}$,
$\mathrm{atol} = 10^{-14}$.

The terminal condition $f_\mathrm{GW} \ge f_\mathrm{ISCO}$ is implemented as an
`events` function, which triggers dense-output interpolation at the exact crossing
rather than relying on a loop check.  The integration time span is set to
$1.05\,t_\mathrm{merge}^\mathrm{0PN}$ to prevent runaway integration.

Output is evaluated on a logarithmically-spaced grid of 10 000 points, so that
early (slow, widely-spaced in $f$) and late (fast, narrowly-spaced in $f$)
evolution are both well-sampled.

- **Implemented in:** `src/smbhb_inspiral/physics.py`, function `integrate_inspiral`
  (line ~332).

---

## References

| Citation | DOI / URL |
|----------|-----------|
| Peters 1964, Phys. Rev. 136, B1224 | https://doi.org/10.1103/PhysRev.136.B1224 |
| Peters & Mathews 1963, Phys. Rev. 131, 435 | https://doi.org/10.1103/PhysRev.131.435 |
| Blanchet 2014, Living Rev. Relativity 17, 2 | https://doi.org/10.12942/lrr-2014-2 |
| Maggiore 2007, *Gravitational Waves: Theory and Experiments* | ISBN 978-0-19-857074-5 |
| Sesana, Vecchio & Colacino 2008, MNRAS 390, 192 | https://doi.org/10.1111/j.1365-2966.2008.13682.x |
| Flanagan & Hughes 1998, Phys. Rev. D 57, 4535 | https://doi.org/10.1103/PhysRevD.57.4535 |
| Robson, Cornish & Liu 2019, CQG 36, 105011 | https://doi.org/10.1088/1361-6382/ab1101 |
| Larson, Hiscock & Hellings 2000, Phys. Rev. D 62, 062001 | https://doi.org/10.1103/PhysRevD.62.062001 |
| Agazie et al. (NANOGrav) 2023, ApJ Lett. 951, L8 | https://doi.org/10.3847/2041-8213/acdac6 |
| Lin, Charisi & Haiman 2026, ApJ 997, 316 | https://doi.org/10.3847/1538-4357/ae29a7 |
| Dormand & Prince 1980, J. Comput. Appl. Math. 6, 19 | https://doi.org/10.1016/0771-050X(80)90013-3 |
