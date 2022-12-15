""" autoregressive random process.
An exemplary application is written in the main() function of this module
References:
    [Kay 1981] S. Kay, "Efficient generation of colored noise", Proceedings IEEE, vol. 69, pp.480-481, April 1981
    [Kay 1988] S. Kay, "Modern Spectral Estimation", Theory & Application, Signal Processing Series, 4th Edition, 1988
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift


def AR_par_est_cov(data: np.array, ar_model_order: int) -> tuple:
    """Estimation of AR parameters using the covariance methode
    (an approximation of a maximum likelihood estimator, see (11.8) and [KAY 1988, pp. 185-189]
    for further details). Note that AT filter parameters may not produce a stable all-pole filter,
    so this method is not recommended for synthesizing an AR time series.

    Args:
        data: array of real data
        ar_model_order: AR model order

    Returns:
        tuple: estimation of ar_filter_parameters containing ar_model_order elements and noise_variance"""
    N = len(data)
    H = np.flipud(data[:ar_model_order])
    # set up H matrix (observation matrix) for least squares estimate
    for i in range(1, N - ar_model_order):
        H = np.vstack((H, np.flipud(data[i:i + ar_model_order])))
    # set up data vector for least squares estimate
    h = data[ar_model_order:N]
    # compute estimate
    ar_filter_parameters = -np.linalg.inv(H.transpose() @ H) @ H.transpose() @ h
    # compute unbiased estimate of excitation white noise variance
    noise_variance = (1 / (N - ar_model_order)) * (h + H @ ar_filter_parameters).transpose() @ (h + H @ ar_filter_parameters)
    return ar_filter_parameters, noise_variance


def get_ar_psd_estimation(data: np.array, ar_model_order: int, n_fft: int, as_dB: bool = True) -> np.array:
    """ AR spectral estimator using the covariance method to estimate the parameters

    Args:
        data: real data
        ar_model_order: target order
        n_fft: number of psd samples
        as_dB: convert to normalized dB?

    Returns: np.array with ar_parameters

    """
    ahat, noise_variance = AR_par_est_cov(data, ar_model_order)
    A = np.zeros(n_fft)
    ar_parameters = np.zeros(n_fft)
    for i, frequency in enumerate(get_frequencies(n_fft)):
        A[i] = 1
        for k in range(ar_model_order):
            A[i] = A[i] + ahat[k] * np.exp(-2j * np.pi * frequency * (k + 1))
        ar_parameters[i] = noise_variance / (abs(A[i]) ** 2)
    if as_dB:
        ar_parameters = convert_to_dB(ar_parameters)  # convert to dB quantities
    return ar_parameters


def get_ar_psd(ar_parameters: np.array, noise_variance: float, n_fft: int, as_dB: bool = True) -> np.array:
    """Compute a set of psd values for the frequency band [-0.5, 0.5],
    given the parameters of an AR model. The FFT is used to evaluate the
    denominator polynominal of the AR psd function.

    Args:
        ar_parameters: model parameter
        noise_variance: variance of excitation noise
        n_fft: number of frequency samples
        as_dB: convert to normalized dB?

    Returns: array of power spectral densities

    """
    denominator = np.append([1], ar_parameters)
    denominator_fft = fftshift(fft(denominator, n_fft))
    ar_psd = noise_variance / (abs(denominator_fft) ** 2)
    if as_dB:
        ar_psd = convert_to_dB(ar_psd)
    return ar_psd


def get_psd_periodogram(data: np.array, len_segment: int, n_fft: int, as_dB: bool = True) -> np.array:
    """Compute the averaged periodogram spectral estimator (as given by (6.24) and (6.25))
    The number of periodograms is the number of whole blocks of length len_segment contained in data.

    Args:
        data: real data
        len_segment: length of single segment
        n_fft: number of frequency samples
        as_dB: convert to normalized dB?

    Returns: averaged periodogram

    """
    segments = [data[_:_ + len_segment] for _ in
                range(0, len(data), len_segment)]
    segment_psds = np.array([abs(fft(seg, n_fft)) ** 2 for seg in segments])
    psd_periodogram = fftshift(np.mean(segment_psds, axis=0))
    if as_dB:
        psd_periodogram = convert_to_dB(psd_periodogram)
    return psd_periodogram


def convert_to_dB(data: np.array) -> np.array:
    """ Normalize data and convert to decibel

    Args:
        data: real data

    Returns: normalized data converted to decibel

    """
    return 10 * np.log10(data / max(data))


def get_frequencies(n_fft):
    """calculate n_fft frequencies inside [-0.5, 0.5] band

    Args:
        n_fft: number of frequency samples

    Returns: frequency array

    """
    return np.arange(n_fft) / n_fft - 0.5


def stepdown(coefficients: np.array, noise_variance: float) -> tuple:
    """Implementation of step-down procedure to find
    the coefficients and prediction error powers for all the
    lower order predictor given the coefficients and prediction error power
    for the model_order linear predictor or equivalently given the filter parameters
    and white noise variance of a model_order AR model.
    See (6.51) and (6.52) in [Kay 1988]

    Args:
        coefficients: real or complex array of parameters (must contain two or more)
        noise_variance: variance of excitation noise

    Returns: tuple containing prediction_coefficients, prediction_error_powers, acf_zero_lag

    """
    model_order = len(coefficients)
    assert model_order >= 2
    prediction_coefficients = np.zeros((model_order, model_order))
    prediction_coefficients[:, -1] = coefficients
    prediction_error_powers = np.zeros(model_order)
    prediction_error_powers[-1] = noise_variance
    #  Begin step-down
    for j in range(model_order - 1):
        k = model_order - j - 1
        denominator = 1 - abs(prediction_coefficients[k, k]) ** 2
        #  Compute lower order prediction error power (6.52)
        prediction_error_powers[k - 1] = prediction_error_powers[k] / denominator
        #  Compute lower order prediction coefficients (6.51)
        k1 = k - 1
        for i in range(k1):
            prediction_coefficients[i, k - 1] = (prediction_coefficients[i, k] - prediction_coefficients[k, k] * np.conj(prediction_coefficients[k - i, k])) / denominator
    # Complete step-down by computing zeroth lag of ACF
    acf_zero_lag = prediction_error_powers[0] / (1 - abs(prediction_coefficients[0, 0]) ** 2)
    return prediction_coefficients, prediction_error_powers, acf_zero_lag


def generate_autoregressive_data(ar_filter_parameters: np.array, noise_variance: float, n_samples: int) -> np.array:
    """Generate a realization of an AR random process
    given the filter parameters and excitation noise variance.
    The starting transient is eliminated because the initial conditions
    of the filter are specified to place the filter output in statistical steady state.
    For further Details see [Kay 1981]

    Args:
        ar_filter_parameters: array of filter parameters
        noise_variance: variance of excitation noise
        n_samples: number of desired data points

    Returns: real data containing n_samples

    """
    model_order = len(ar_filter_parameters)
    noise = np.random.randn(n_samples)
    data = np.zeros(n_samples)
    aa, rho, rho0 = get_initial_ar_filter_conditions(ar_filter_parameters, noise_variance)
    data[0] = np.sqrt(rho0) * noise[0]
    for k in range(1, model_order):
        data[k] = np.sqrt(rho[k - 1]) * noise[k]
        for ll in range(k - 1):
            data[k] = data[k] - aa[ll, k - 1] * data[k - ll]
    # generate remainder of AR data
    for k in range(model_order, n_samples):
        data[k] = np.sqrt(noise_variance) * noise[k]
        for ll in range(model_order):
            data[k] = data[k] - ar_filter_parameters[ll] * data[k - ll - 1]
    return data


def get_initial_ar_filter_conditions(ar_filter_parameters: np.array, noise_variance: float) -> tuple:
    """ initial filter conditions. For more than 1 coefficient, use step-down method.

    Args:
        ar_filter_parameters: array of filter parameters
        noise_variance: variance of excitation noise

    Returns: tuple containing prediction_coefficients, prediction_error_powers, acf_zero_lag

    """
    if len(ar_filter_parameters) > 1:
        return stepdown(ar_filter_parameters, noise_variance)
    return None, None, noise_variance / (1 - ar_filter_parameters[0] ** 2)


def main():
    n_samples = 32
    ar_filter_parameters = [0, 0.9025]
    noise_variance = 1
    n_fft = 1024
    n_estimated_coefficients = 2

    data = generate_autoregressive_data(ar_filter_parameters, noise_variance, n_samples)
    frequencies = get_frequencies(n_fft)
    periodogram_psd = get_psd_periodogram(data, n_samples, n_fft)
    averaged_periodogram_psd = get_psd_periodogram(data, n_samples // 4, n_fft)
    ar_psd = get_ar_psd(ar_filter_parameters, noise_variance, n_fft)
    psd_cov_dB = get_ar_psd_estimation(data, n_estimated_coefficients, n_fft)

    plt.plot(frequencies, periodogram_psd, label="Standard")
    plt.plot(frequencies, averaged_periodogram_psd, label="Bartlett")
    plt.plot(frequencies, ar_psd, label="True")
    plt.plot(frequencies, psd_cov_dB, label="AR covariant")
    plt.legend()
    plt.xlabel("Frequency (arb. units)")
    plt.ylabel("Intensity (dB)")
    plt.show()


if __name__ == '__main__':
    main()
