import soundfile as sf
import numpy as np
from vnsim import calc_vnsim
from argparse import ArgumentParser
from pathlib import Path
import warnings


def load_and_preprocess_signals(ref_path, deg_path):
    """
    Load and pre-process reference and degraded audio signals.

    Args:
        ref_path (Path): Path to the reference audio file.
        deg_path (Path): Path to the degraded audio file.

    Returns:
        tuple: Processed reference signal, degraded signal, sample rate, and number of channels.
    """

    # reading reference and degraded audio files
    ref_sig, sample_rate = sf.read(str(ref_path))
    deg_sig, _ = sf.read(str(deg_path))

    n_channels_ref = ref_sig.shape[1]
    n_channels_deg = deg_sig.shape[1]

    # Number of samples to append
    num_zeros = 11520

    # Create an array of zeros with shape (11520, number of channels)
    zeros_ref = np.zeros((num_zeros, n_channels_ref))
    zeros_deg = np.zeros((num_zeros, n_channels_deg))

    # Concatenate the zeros array with the original signals
    ref_sig = np.vstack((zeros_ref, ref_sig))
    deg_sig = np.vstack((zeros_deg, deg_sig))

    return ref_sig, deg_sig, sample_rate, n_channels_deg


def calculate_ambiqual(ref_path, deg_path, intensity_threshold, elc, ignore_freq_bands):
    """
    Calculate the Ambiqual metrics for the given audio files.

    Args:
        ref_path (Path): Path to the reference audio file.
        deg_path (Path): Path to the degraded audio file.
        intensity_threshold (int): Intensity threshold for NSIM calculation.
        elc (int): Equal loudness contour adjustment parameter:
            0 - no elc
            1 - elc by boosting low and high frequencies
            2 - elc by attenuating low and high frequencies
        ignore_freq_bands (int): ignoring high frequency bands (0:32):
            0 - all 32 frequency bands are taken into account
            k - k-th to 32 frequency bands are ignored in calculations

    Returns:
        tuple: List of NSIM values, LQ, LA values.
    """

    ref_sig, deg_sig, sample_rate, n_channels = load_and_preprocess_signals(ref_path, deg_path)

    nsim_values = []
    nsim_values_nan = []

    alpha = 0.999
    beta = 0.034
    gamma = 0.078
    delta = 0.001
    epsilon = 0.001
    zeta = 0.001
    chi = 0.095
    psi = 0.135
    omega = 0.174

    for i in range(16):
        if i >= n_channels:
            vnsim = np.nan

        else:
            vnsim = calc_vnsim(ref_sig[:, i], deg_sig[:, i],
                               sample_rate, intensity_threshold, elc, ignore_freq_bands)

        nsim_values_nan.append(vnsim)
        #print(f"vnsim_{i}:", round(vnsim, 6))

    LQ = nsim_values_nan[0]

    nsim_values = []
    for i in range(16):
        if np.isnan(nsim_values_nan[i]):
            nsim_values.append(0.1)
        else:
            nsim_values.append(nsim_values_nan[i])

    LA = (
            (nsim_values[1] ** alpha) * (nsim_values[3] ** alpha) *
            (nsim_values[4] ** beta) * (nsim_values[8] ** beta) *
            (nsim_values[9] ** gamma) * (nsim_values[15] ** gamma) *
            (nsim_values[5] ** delta) * (nsim_values[7] ** delta) *
            (nsim_values[10] ** epsilon) * (nsim_values[14] ** epsilon) *
            (nsim_values[11] ** zeta) * (nsim_values[13] ** zeta) *
            (nsim_values[2] ** chi) * (nsim_values[6] ** psi) *
            (nsim_values[2] ** omega)
    )

    return nsim_values_nan, LQ, LA


def parse_args():
    parser = ArgumentParser("Ambiqual")

    parser.add_argument("--ref",
                        type=Path,
                        help="Path to reference audio file.",
                        required=True
                        )

    parser.add_argument("--deg",
                        type=Path,
                        help="Path to degraded audio file.",
                        required=True
                        )

    parser.add_argument("--level",
                        type=float,
                        help="Intensity threshold for NSIM calculation.",
                        required=False
                        )

    parser.add_argument("--elc",
                        type=int,
                        help="",
                        required=False
                        )

    parser.add_argument("--ignorefreqbands",
                        type=int,
                        help="Number of frequency bands to ignore.",
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    ref_path = args.ref
    deg_path = args.deg
    intensity_threshold = args.level
    elc = args.elc
    ignore_freq_bands = args.ignorefreqbands

    if intensity_threshold == None:
        intensity_threshold = -180

    if elc == None:
        elc = 0

    if ignore_freq_bands == None:
        ignore_freq_bands = 0

    nsim_values, LQ, LA = calculate_ambiqual(ref_path,
                                             deg_path,
                                             intensity_threshold,
                                             elc,
                                             ignore_freq_bands
                                            )

    # print("vnsim", nsim_values)
    print("")
    print("LQ: ", LQ)
    print("LA: ", LA)
