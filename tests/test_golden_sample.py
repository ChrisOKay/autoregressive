import pickle

import numpy as np

from exercise import generate_autoregressive_data, get_frequencies, get_psd_periodogram, get_ar_psd, get_ar_psd_estimation


def test_golden_sample():

    with open('golden_sample.pickle', 'rb') as handle:
        (gs_fixed_noise,gs_periodogram_psd,gs_ar_psd,gs_PSD_cov_dB) = pickle.load(handle)

    N = 32
    a = [0, 0.9025]
    sig2u = 1
    x = generate_autoregressive_data(a, sig2u, N, gs_fixed_noise)
    # plt.plot(x, '.-')
    # plt.show()
    n_fft = 1024
    frequencies = get_frequencies(n_fft)
    periodogram_psd = get_psd_periodogram(x, N, n_fft)
    ar_psd = get_ar_psd(a, sig2u, n_fft)
    p = 2
    PSD_cov_dB = get_ar_psd_estimation(x, p, n_fft)

    assert np.array_equal(periodogram_psd, gs_periodogram_psd)
    assert np.array_equal(ar_psd, gs_ar_psd)
    assert np.array_equal(PSD_cov_dB, gs_PSD_cov_dB)