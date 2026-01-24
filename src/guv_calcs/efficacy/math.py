import math


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
    return (k1 * (1 - f) + k2 - k2 * (1 - f)) * irrad * 3.6


def seconds_to_S(S, irrad, k1, k2=0, f=0, tol=1e-10, max_iter=100):
    """
    Calculate time in seconds to reach survival fraction S.

    S: float, (0,1) - surviving fraction
    irrad: float or list - fluence/irradiance in uW/cm2
    k1: float or list - first susceptibility value, cm2/mJ
    k2: float or list - second susceptibility value, cm2/mJ
    f: float or list - (0,1) resistant fraction
    tol: float, numerical tolerance
    max_iter: maximum number of iterations to wait for solution to converge

    For multi-wavelength: pass lists for irrad, k1, k2, f (same length).
    The k*irrad values are summed, and f values are averaged.
    """
    # Handle multi-wavelength case
    # Note: divide irrad by 1000 to convert µW/cm² to mW/cm² (k is in cm²/mJ)
    if isinstance(irrad, (list, tuple)):
        k1_irrad = sum(k * i / 1000 for k, i in zip(k1, irrad))
        k2_irrad = sum(k * i / 1000 for k, i in zip(k2, irrad))
        f_eff = sum(f) / len(f)
    else:
        k1_irrad = k1 * irrad / 1000
        k2_irrad = k2 * irrad / 1000
        f_eff = f

    def S_of_t(t):
        return (1 - f_eff) * math.exp(-k1_irrad * t) + f_eff * math.exp(-k2_irrad * t)

    # Bracket the root
    t_low = 0.0
    t_high = 1.0
    while S_of_t(t_high) > S:
        t_high *= 2.0
    # Bisection
    for _ in range(max_iter):
        t_mid = 0.5 * (t_low + t_high)
        if S_of_t(t_mid) > S:
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
