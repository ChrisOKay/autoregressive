import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift


def AR_par_est_cov(x, p):
    N = len(x)
    for i in range(
            N - p):  # set up H matrix (observation matrix) for least squares estimate
        if i == 0:
            H = np.flipud(x[:p])
        else:
            H = np.vstack((H, np.flipud(x[i:i + p])))
    h = x[p:N];  # set up data vector for least squares estimate
    ahat = -np.linalg.inv(H.transpose() @ H) @ H.transpose() @ h  # compute estimate
    # compute unbiased estimate of excitation white noise variance
    sig2uhat = (1 / (N - p)) * (h + H @ ahat).transpose() @ (h + H @ ahat);
    return ahat, sig2uhat


def PSD_est_AR_cov(x, p, n_fft, as_dB=True):
    # estimate AR parameters using the covariance method
    ahat, sig2uhat = AR_par_est_cov(x, p)
    A = np.zeros(n_fft).astype(complex)
    PAR = np.zeros(n_fft).astype(complex)
    for i, frequency in enumerate(get_frequencies(n_fft)):
        A[i] = 1
        for k in range(p):
            A[i] = A[i] + ahat[k] * np.exp(-2j * np.pi * frequency * (k+1))
        PAR[i] = sig2uhat / (abs(A[i]) ** 2)
    if as_dB:
        PAR = convert_to_dB(PAR)  # convert to dB quantities
    return PAR


def get_ar_psd(coefficients, noise_variance, n_fft, as_dB=True):
    """ Compute denominator frequency function """
    denominator = np.append([1], coefficients)
    denominator_fft = fftshift(fft(denominator, n_fft))
    ar_psd = noise_variance / (abs(denominator_fft) ** 2)
    if as_dB:
        ar_psd = convert_to_dB(ar_psd)
    return ar_psd


def get_psd_periodogram(data, len_segment, n_fft, as_dB=True):
    segments = [data[_:_ + len_segment] for _ in
                range(0, len(data), len_segment)]
    segment_psds = np.array([abs(fft(seg, n_fft)) ** 2 for seg in segments])
    psd_periodogram = fftshift(np.mean(segment_psds, axis=0))
    if as_dB:
        psd_periodogram = convert_to_dB(psd_periodogram)
    return psd_periodogram


def convert_to_dB(data):
    return 10 * np.log10(data / max(data))


def get_frequencies(n_fft):
    return np.arange(n_fft) / n_fft - 0.5


def stepdown(model_order, coefficients, noise_variance):
    aa = np.zeros((model_order, model_order))
    aa[:, -1] = coefficients
    rho = np.zeros(model_order)
    rho[-1] = noise_variance
    #  Begin step-down
    for j in range(model_order - 1):
        k = model_order - j - 1
        den = 1 - abs(aa[k, k]) ** 2
        #  Compute lower order prediction error power (6.52)
        rho[k - 1] = rho[k] / den
        #  Compute lower order prediction coefficients (6.51)
        k1 = k - 1
        for i in range(k1):
            aa[i, k - 1] = (aa[i, k] - aa[k, k] * np.conj(aa[k - i, k])) / den
    # Complete step-down by computing zeroth lag of ACF
    rho0 = rho[0] / (1 - abs(aa[0, 0]) ** 2)
    return aa, rho, rho0


def generate_autoregressive_data(coefficients, noise_variance, n_samples):
    model_order = len(coefficients)  # get AR model order
    noise = np.random.randn(n_samples)
    data = np.zeros(n_samples)
    aa, rho, rho0 = get_initial_ar_filter_conditions(coefficients, model_order,
                                                     noise_variance)
    data[0] = np.sqrt(rho0) * noise[0]
    for k in range(1, model_order):
        data[k] = np.sqrt(rho[k - 1]) * noise[k]
        for ll in range(k - 1):
            data[k] = data[k] - aa[ll, k - 1] * data[k - ll]
    # %   generate remainder of AR data
    for k in range(model_order, n_samples):
        data[k] = np.sqrt(noise_variance) * noise[k]
        for ll in range(model_order):
            data[k] = data[k] - coefficients[ll] * data[k - ll - 1]
    return data


def get_initial_ar_filter_conditions(coefficients, model_order,
                                     noise_variance):
    if model_order > 1:
        return stepdown(model_order, coefficients, noise_variance)
    return None, None, noise_variance / (1 - coefficients[0] ** 2)


if __name__ == '__main__':
    N = 32
    a = [0, 0.9025]
    sig2u = 1
    x = generate_autoregressive_data(a, sig2u, N)
    # plt.plot(x, '.-')
    # plt.show()
    n_fft = 1024
    frequencies = get_frequencies(n_fft)
    periodogram_psd = get_psd_periodogram(x, N, n_fft)
    plt.plot(frequencies, periodogram_psd, label="Standard")

    plt.plot(frequencies, get_psd_periodogram(x, N//4, n_fft), label="Bartlett")
    ar_psd = get_ar_psd(a, sig2u, n_fft)
    plt.plot(frequencies, ar_psd, label="True")
    p = 2
    PSD_cov_dB = PSD_est_AR_cov(x, p, n_fft)
    plt.plot(frequencies, PSD_cov_dB, label="AR covariant")
    plt.legend()
    plt.xlabel("Frequency (arb. units)")
    plt.ylabel("Intensity (dB)")
    plt.show()
