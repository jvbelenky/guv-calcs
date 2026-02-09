import numpy as np


def CADR_CFM(cubic_feet, irrad, k1, k2=0, f=0):
    """Calculate clean air delivery rate in cubic feet per minute."""
    return eACH_UV(irrad=irrad, k1=k1, k2=k2, f=f) * cubic_feet / 60


def CADR_LPS(cubic_meters, irrad, k1, k2=0, f=0):
    """Calculate clean air delivery rate in liters per second."""
    return eACH_UV(irrad=irrad, k1=k1, k2=k2, f=f) * cubic_meters * 1000 / 60 / 60


def eACH_UV(irrad, k1, k2=0, f=0):
    """
    Calculate equivalent air changes per hour from UV.

    For multi-wavelength: pass lists for irrad, k1, k2, f (same length).
    The eACH values are additive across wavelengths.
    """
    if isinstance(irrad, (list, tuple)):
        return sum(eACH_UV(i, k, kk, ff) for i, k, kk, ff in zip(irrad, k1, k2, f))
    return (k1 * (1 - f) + k2 * f) * irrad * 3.6


def survival_fraction(t, irrad, k1, k2=0, f=0):
    """Survival fraction S(t) for the biphasic inactivation model."""
    t = np.asarray(t)
    if isinstance(irrad, (list, tuple)):
        k1_irrad = sum(k * i / 1000 for k, i in zip(k1, irrad))
        k2_irrad = sum(k * i / 1000 for k, i in zip(k2, irrad))
        f_eff = sum(f) / len(f)
    else:
        k1_irrad = k1 * irrad / 1000
        k2_irrad = k2 * irrad / 1000
        f_eff = f
    return (1 - f_eff) * np.exp(-k1_irrad * t) + f_eff * np.exp(-k2_irrad * t)


def seconds_to_S(S, irrad, k1, k2=0, f=0, tol=1e-10, max_iter=100):
    """Time in seconds to reach survival fraction S (bisection on survival_fraction)."""
    t_low = 0.0
    t_high = 1.0
    while float(survival_fraction(t_high, irrad, k1, k2, f)) > S:
        t_high *= 2.0
    for _ in range(max_iter):
        t_mid = 0.5 * (t_low + t_high)
        if float(survival_fraction(t_mid, irrad, k1, k2, f)) > S:
            t_low = t_mid
        else:
            t_high = t_mid
        if abs(t_high - t_low) < tol:
            break
    return 0.5 * (t_low + t_high)


# Log reduction functions - wrappers for seconds_to_S with preset survival fractions
def log1(irrad, k1, k2=0, f=0, **kwargs):
    """Time to 90% inactivation (1-log reduction)."""
    return seconds_to_S(0.1, irrad, k1, k2, f, **kwargs)


def log2(irrad, k1, k2=0, f=0, **kwargs):
    """Time to 99% inactivation (2-log reduction)."""
    return seconds_to_S(0.01, irrad, k1, k2, f, **kwargs)


def log3(irrad, k1, k2=0, f=0, **kwargs):
    """Time to 99.9% inactivation (3-log reduction)."""
    return seconds_to_S(0.001, irrad, k1, k2, f, **kwargs)


def log4(irrad, k1, k2=0, f=0, **kwargs):
    """Time to 99.99% inactivation (4-log reduction)."""
    return seconds_to_S(0.0001, irrad, k1, k2, f, **kwargs)


def log5(irrad, k1, k2=0, f=0, **kwargs):
    """Time to 99.999% inactivation (5-log reduction)."""
    return seconds_to_S(0.00001, irrad, k1, k2, f, **kwargs)
