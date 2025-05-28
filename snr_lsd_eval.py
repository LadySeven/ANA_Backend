import numpy as np


def computer_snr_from_estimate(signal):
    signal_power = np.mean(signal ** 2)
    noise_estimate = np.median(np.abs(signal))
    noise_power = noise_estimate ** 2 if noise_estimate > 0 else 1e-12

    if signal_power <= 0:
        return -np.inf

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def computer_lsd(system_spectrum, reference_spectrum):

    if np.array_equal(system_spectrum, reference_spectrum):
        return 0.0

    assert system_spectrum.shape == reference_spectrum.shape, "Spectra must match in shape"

    log_diff = 10 * np.log10(np.maximum(system_spectrum, 1e-10) / np.maximum(reference_spectrum, 1e-10))

    lsd = np.sqrt(np.mean(log_diff ** 2))
    return lsd
