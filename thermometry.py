import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

# from https://github.com/gold-standard-phantoms/mr-image-tools/blob/main/src/mrimagetools/filters/multiecho_thermometry_filter.py

GAMMA_H = 42.57747892e6  # Hz/T

def lsq_fit_thermometry_signal_model(
    echo_times: NDArray[np.floating],
    signal_values: NDArray[np.floating],
    initial_guess: list[float],
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    r"""Least-squares fit of `thermometry_signal_model` to a single signal vector.

    This is a bounded non-linear least squares fit using `scipy.optimize.curve_fit`.
    If the optimizer fails to converge, NaNs are returned for both the parameters
    and their covariance.

    Args:
        echo_times: Echo times (seconds), shape `(n_echoes,)`.
        signal_values: Magnitude signal values, shape `(n_echoes,)`.
        initial_guess: Initial parameter guess as
            `[A1, A2, R2*_1, R2*_2, df, dphi_deg]`.

    Returns:
        `(popt, pcov, r_squared)` where:

        - `popt`: Optimal parameters, shape `(6,)`.
        - `pcov`: Estimated covariance matrix, shape `(6, 6)`.
        - `r_squared`: Coefficient of determination $R^2$ for the fitted model.
    """
    max_amplitude = np.max(signal_values)
    bounds = (
        [0, 0, 1e-3, 1e-3, 0, -360],
        [10 * max_amplitude, 10 * max_amplitude, 1000, 1000, 1000, 360],
    )
    try:
        popt, pcov, *_ = curve_fit(
            thermometry_signal_model,
            echo_times,
            signal_values,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000,
        )
    except RuntimeError:
        popt = np.array([np.nan] * len(initial_guess), dtype=np.float64)
        pcov = np.full((len(initial_guess), len(initial_guess)), np.nan)

    r_squared = calculate_r_squared(
        signal_values, thermometry_signal_model(echo_times, *popt)
    )
    return popt, pcov, r_squared

def thermometry_signal_model(
    t: NDArray[np.floating],
    amplitude_1: float,
    amplitude_2: float,
    r2star_1: float,
    r2star_2: float,
    df: NDArray[np.floating],
    dphi_deg: float,
) -> NDArray[np.floating]:
    r"""Evaluate the dual-resonance magnitude signal model.

    The magnitude signal is modelled as:

    $$S(t)=\sqrt{
      A_1^2 e^{-2R_{2,1}^* t}
      + A_2^2 e^{-2R_{2,2}^* t}
      + 2A_1A_2 e^{-(R_{2,1}^*+R_{2,2}^*) t}\cos(2\pi\,\Delta f\,t+\Delta\phi)
    }$$

    Args:
        t: Time vector (seconds), typically the echo times.
        amplitude_1: Amplitude of resonance component 1.
        amplitude_2: Amplitude of resonance component 2.
        r2star_1: $R_2^*$ of component 1 (1/s).
        r2star_2: $R_2^*$ of component 2 (1/s).
        df: Frequency difference $\Delta f$ (Hz). Can be scalar or vector.
        dphi_deg: Phase offset $\Delta\phi$ (degrees).

    Returns:
        Modelled magnitude signal $S(t)$.
    """
    dphi_rad = np.deg2rad(dphi_deg)
    radicand: NDArray[np.float64] = (
        amplitude_1**2 * np.exp(-2 * r2star_1 * t)
        + amplitude_2**2 * np.exp(-2 * r2star_2 * t)
        + 2
        * amplitude_1
        * amplitude_2
        * np.exp(-(r2star_1 + r2star_2) * t)
        * np.cos(2 * np.pi * df * t + dphi_rad)
    )
    # Prevent negative values under the square root (numerical precision).
    np.maximum(radicand, 0.0, out=radicand)
    return np.sqrt(radicand)

def calculate_temperature_from_df(
    df: NDArray[np.floating], magnetic_field_tesla: float
) -> NDArray[np.floating]:
    r"""Convert frequency difference $\Delta f$ (Hz) to temperature (°C).

    Uses the calibration:

    $$T[^\circ C] = 193.35 - 1.02\times10^8 \cdot \frac{|\Delta f|\,[\mathrm{Hz}]}{\gamma B_0}$$

    Note:
        The implementation uses $|\Delta f|$ to return a non-negative temperature
        shift with respect to the calibration.

    Args:
        df: Frequency difference $\Delta f$ in Hz.
        magnetic_field_tesla: Magnetic field strength $B_0$ in Tesla.

    Returns:
        Temperature in °C (same shape as `df`).
    """
    return 193.35 - (1.02e8 * np.abs(df)) / (GAMMA_H * magnetic_field_tesla)

def calculate_r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Calculate coefficient of determination.

    :param observed: Measured signal values
    :param predicted: Model-predicted signal values
    :return: R² value between 0 and 1

    .. math::
        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res} = \\sum_i (y_i - \\hat{y}_i)^2` and
    :math:`SS_{tot} = \\sum_i (y_i - \\bar{y})^2`
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
