"""Ozone generation estimation from UV lamp spectra.

Physics: photons with λ < 242nm dissociate O2 into two O atoms, each of
which quickly combines with O2 to form O3 (quantum yield ≈ 2).

The generation constant C relates average UV fluence rate to steady-state
ozone production rate:

    ozone_rate [ppb/hr] = C × avg_fluence [µW/cm²]

And the steady-state ozone increase is:

    Δ[O3] [ppb] = C × avg_fluence / (ACH + k_decay)

C is computed by integrating the lamp spectrum against O2 absorption
cross-sections weighted by photon energy.
"""

import numpy as np

# ---------------------------------------------------------------------------
# O2 absorption cross-sections (185-245 nm) at room temperature.
#
# Sources:
#   185-202 nm: Yoshino et al. (1992), 1nm-averaged Schumann-Runge band
#               cross-sections at 300K. These include both the S-R band
#               rotational structure and the underlying Herzberg continuum.
#               Planet. Space Sci. 40(2), 185-192.
#   203-204 nm: MPI-Mainz UV/VIS Spectral Atlas (Herzberg continuum)
#   205-245 nm: JPL "Chemical Kinetics and Photochemical Data for Use
#               in Stratospheric Modeling" Evaluation Number 12
#
# Units: [wavelength_nm, cross_section_cm2]
# ---------------------------------------------------------------------------
_O2_DATA = np.array([
    [185, 6.229e-21],
    [186, 8.913e-21],
    [187, 1.785e-21],
    [188, 4.387e-21],
    [189, 5.330e-22],
    [190, 2.253e-21],
    [191, 2.530e-22],
    [192, 7.700e-22],
    [193, 1.577e-22],
    [194, 1.710e-22],
    [195, 1.146e-22],
    [196, 3.308e-23],
    [197, 5.532e-23],
    [198, 2.142e-23],
    [199, 1.322e-23],
    [200, 1.426e-23],
    [201, 9.211e-24],
    [202, 7.205e-24],
    [203, 7.765e-24],
    [204, 7.664e-24],
    [205, 7.350e-24],
    [206, 7.130e-24],
    [207, 7.050e-24],
    [208, 6.860e-24],
    [209, 6.680e-24],
    [210, 6.510e-24],
    [211, 6.240e-24],
    [212, 6.050e-24],
    [213, 5.890e-24],
    [214, 5.720e-24],
    [215, 5.590e-24],
    [216, 5.350e-24],
    [217, 5.130e-24],
    [218, 4.880e-24],
    [219, 4.640e-24],
    [220, 4.460e-24],
    [221, 4.260e-24],
    [222, 4.090e-24],
    [223, 3.890e-24],
    [224, 3.670e-24],
    [225, 3.450e-24],
    [226, 3.210e-24],
    [227, 2.980e-24],
    [228, 2.770e-24],
    [229, 2.630e-24],
    [230, 2.430e-24],
    [231, 2.250e-24],
    [232, 2.100e-24],
    [233, 1.940e-24],
    [234, 1.780e-24],
    [235, 1.630e-24],
    [236, 1.480e-24],
    [237, 1.340e-24],
    [238, 1.220e-24],
    [239, 1.100e-24],
    [240, 1.010e-24],
    [241, 8.800e-25],
    [242, 8.100e-25],
    [243, 3.900e-25],
    [244, 1.300e-25],
    [245, 5.000e-26],
])

_O2_WAVELENGTHS = _O2_DATA[:, 0]
_O2_CROSS_SECTIONS = _O2_DATA[:, 1]

# Physical constants
_H = 6.62607015e-34    # Planck constant [J·s]
_C = 2.99792458e8      # speed of light [m/s]
_HC = _H * _C          # [J·m]
_K_B = 1.380649e-23    # Boltzmann constant [J/K]
_F_O2 = 0.2095         # O2 mole fraction in dry air
_O2_DISSOCIATION_NM = 242.4  # O2 dissociation wavelength limit [nm]


def ozone_generation_constant(spectrum, T=293.15, P=101325.0):
    """Compute the ozone generation constant C from a lamp spectrum.

    C is defined such that:
        ozone_rate [ppb/hr] = C × avg_fluence_rate [µW/cm²]

    Parameters
    ----------
    spectrum : Spectrum
        Lamp emission spectrum (wavelengths in nm, relative intensities).
    T : float
        Temperature [K]. Default 293.15 K (20°C).
    P : float
        Pressure [Pa]. Default 101325 Pa (1 atm).

    Returns
    -------
    float
        Generation constant C [ppb/hr per µW/cm²].
        Returns 0.0 if spectrum has no emission below 242 nm.
    """
    wv = np.array(spectrum.wavelengths)
    ints = np.array(spectrum.intensities)

    # clip negative intensities (measurement noise)
    ints = np.clip(ints, 0, None)

    # normalize spectrum so ∫S(λ)dλ = 1
    total = np.trapz(ints, wv)
    if total <= 0:
        return 0.0
    s_norm = ints / total

    # interpolate O2 cross-sections onto spectrum wavelength grid in log space
    # (cross-sections span orders of magnitude, so log-interp is appropriate)
    log_sigma = np.interp(wv, _O2_WAVELENGTHS, np.log(_O2_CROSS_SECTIONS),
                          left=-np.inf, right=-np.inf)
    sigma = np.where(np.isfinite(log_sigma), np.exp(log_sigma), 0.0)

    # spectral integral: ∫ σ(λ) × S_norm(λ) × λ dλ  [cm² · nm]
    integrand = sigma * s_norm * wv
    spec_integral = np.trapz(integrand, wv)

    if spec_integral <= 0:
        return 0.0

    # number densities [cm⁻³]
    n_air = P / (_K_B * T) * 1e-6   # m⁻³ → cm⁻³
    n_o2 = _F_O2 * n_air

    # C = 2 × n_O2 × (1e-15 / hc) × (3600 / (n_air × 1e-9)) × spectral_integral
    #
    # Factor breakdown:
    #   2        — quantum yield (each O2 photolysis → 2 O3)
    #   1e-6     — µW → W
    #   1e-9     — nm → m (wavelength in photon energy hc/λ)
    #   1e-15    — combined µW·nm → W·m
    #   3600     — s → hr
    #   1/(n_air × 1e-9) — molecules/cm³ → ppb
    C = 2 * n_o2 * (1e-15 / _HC) * (3600 / (n_air * 1e-9)) * spec_integral

    return float(C)
