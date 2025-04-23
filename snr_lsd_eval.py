import numpy as np

def computer_snr (signal, noise):
    signal_power = np.mean (signal ** 2)
    noise_power =np.mean (noise ** 2)

    snr = 10 * np.log10 (signal_power / noise_power)

    return snr

def computer_lsd (system_spectrum, reference_spectrum):
    assert system_spectrum.shape == reference_spectrum.shape, "Spectra must be the same shape"

    log_diff = 10 * np.log10(system_spectrum / reference_spectrum)
    lsd = np.sqrt(np.mean(log_diff ** 2))

    return lsd
