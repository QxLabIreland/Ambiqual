from ambiqual import calculate_ambiqual
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def validate_ambiqual(ref_files_lst, deg_files_lst, path, intensity_threshold, elc, ignorefreqbands, experiment):

    ref_lst = []
    deg_lst = []
    nsims_lst = []
    LQ_lst = []
    LA_lst = []

    start = 0
    for i in range(len(ref_files_lst)):
        for j in range(start, start + int(len(deg_files_lst)/len(ref_files_lst))):
            nsims, LQ, LA = calculate_ambiqual(path + ref_files_lst[i],
                                               path + deg_files_lst[j],
                                               intensity_threshold,
                                               elc,
                                               ignorefreqbands)

            ref_lst.append(ref_files_lst[i])
            deg_lst.append(deg_files_lst[j])
            nsims_lst.append(nsims)
            LQ_lst.append(LQ)
            LA_lst.append(LA)

            start += 1

    df = pd.DataFrame({
        'filename': deg_lst,
        "intensity_threshold": intensity_threshold,
        "elc": elc,
        "ignore_freq_bands": ignorefreqbands,
        'LQ': LQ_lst,
        'LA': LA_lst
    })

    nsims_columns = [f'nsim_{i}' for i in range(len(nsims_lst[0]))]

    nsims_df = pd.DataFrame(nsims_lst, columns=nsims_columns)

    # Concatenate the two DataFrames
    df = pd.concat([df, nsims_df], axis=1)

    df.to_csv("validation_experiment_" + str(experiment) + ".csv", index=False, na_rep='NaN')


def plot_lq():

    mushra_df = pd.read_csv("MUSHRA_experiment_1.csv")
    ambiqual_df = pd.read_csv("validation_experiment_1.csv")

    LQ_values = ambiqual_df["LQ"].values
    LQ_array = LQ_values.reshape(6, 4).T

    mushra_results_df = mushra_df.iloc[45:, :]

    mushra_results = mushra_results_df["mean"].values
    sample_results_by_encoding_LQ = mushra_results.reshape(6, 5).T


    labels_df = mushra_results_df["srcdir"]
    labels = labels_df[0::5]

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  # 1 row, 2 columns

    # First subplot
    ax1.plot(sample_results_by_encoding_LQ[1, :], LQ_array[0, :], 'o', markersize=14, linewidth=4, label='3OA512', markerfacecolor='none', markeredgewidth=2.5)
    ax1.plot(sample_results_by_encoding_LQ[2, :], LQ_array[1, :], 'd', markersize=14, linewidth=4, label='3OA256', markerfacecolor='none', markeredgewidth=2.5)
    ax1.plot(sample_results_by_encoding_LQ[3, :], LQ_array[2, :], '>', markersize=14, linewidth=4, label='FOA128', markerfacecolor='none', markeredgewidth=2.5)
    ax1.plot(sample_results_by_encoding_LQ[4, :], LQ_array[3, :], 's', markersize=14, linewidth=4, label='FOA32', markerfacecolor='none', markeredgewidth=2.5)

    ax1.axis([0, 100, 0, 1])
    ax1.tick_params(labelsize=20)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=15)
    ax1.set_title('LQ (single source)', fontsize=20, fontweight='bold')
    ax1.set_xlabel('MUSHRA scores', fontsize=20)
    ax1.set_ylabel('LQ (NSIM)', fontsize=20)


    # Second subplot
    ax2.plot(sample_results_by_encoding_LQ[1:, 0], LQ_array[0:, 0], '*', markersize=14, linewidth=4, markerfacecolor='none', markeredgewidth=2.5)
    ax2.plot(sample_results_by_encoding_LQ[1:, 1], LQ_array[0:, 1], 's', markersize=14, linewidth=4, markerfacecolor='none', markeredgewidth=2.5)
    ax2.plot(sample_results_by_encoding_LQ[1:, 2], LQ_array[0:, 2], 'd', markersize=14, linewidth=4, markerfacecolor='none', markeredgewidth=2.5)
    ax2.plot(sample_results_by_encoding_LQ[1:, 3], LQ_array[0:, 3], '*', markersize=14, linewidth=4, markerfacecolor='none', markeredgewidth=2.5)
    ax2.plot(sample_results_by_encoding_LQ[1:, 4], LQ_array[0:, 4], 's', markersize=14, linewidth=4, markerfacecolor='none', markeredgewidth=2.5)
    ax2.plot(sample_results_by_encoding_LQ[1:, 5], LQ_array[0:, 5], '^', markersize=14, linewidth=4, markerfacecolor='none', markeredgewidth=2.5)

    ax2.axis([0, 100, 0, 1])
    ax2.tick_params(labelsize=20)
    ax2.grid(True)
    ax2.legend(labels, loc='lower right', fontsize=14)
    ax2.set_title('LQ (single source)', fontsize=20,  fontweight='bold')
    ax2.set_xlabel('MUSHRA scores', fontsize=20)
    ax2.set_ylabel('LQ (NSIM)', fontsize=20)

    # Adjust the layout and show the figure
    plt.tight_layout()
    plt.show()

    # Save the figure as a PDF
    fig.savefig("LQ.pdf", bbox_inches='tight')


def plot_la():

    # Load the data for plots 1 and 2
    mushra_df = pd.read_csv("MUSHRA_experiment_1.csv")
    ambiqual_df = pd.read_csv("validation_experiment_1.csv")

    LA_values = ambiqual_df["LA"].values
    LA_array = LA_values.reshape(6, 4).T

    #local_quality_results = mushra_df.iloc[0:45, :]
    results_df = mushra_df.iloc[45:, :]

    #quality_results = local_quality_results["mean"].values
    accuracy_results = results_df["mean"].values
    sample_results_by_encoding_LA = accuracy_results.reshape(6, 5).T

    # Load the data for plots 3 and 4
    mushra_multi_df = pd.read_csv("MUSHRA_experiment_2.csv")
    ambiqual_multi_df = pd.read_csv("validation_experiment_2.csv")

    mushra_multi_mean = mushra_multi_df.iloc[:, 3:].mean(axis=1, skipna=True).tolist()
    mushra_multi_mean = np.reshape(np.reshape(mushra_multi_mean, (6, 8)).T[1:, :].flatten(), (7, 6))

    LA_multi_values = ambiqual_multi_df["LA"].values
    LA_multi = LA_multi_values.reshape(6, 7).T

    # Create a 1x4 grid for the subplots
    fig, axs = plt.subplots(1, 4, figsize=(32, 7))  # 1 row, 4 columns

    # Plot 1
    axs[0].plot(sample_results_by_encoding_LA[1, :], LA_array[0, :], 'o', markersize=14,  label='3OA512', markerfacecolor='none', markeredgewidth=2.5)
    axs[0].plot(sample_results_by_encoding_LA[2, :], LA_array[1, :], 'd', markersize=14,  label='3OA256', markerfacecolor='none', markeredgewidth=2.5)
    axs[0].plot(sample_results_by_encoding_LA[3, :], LA_array[2, :], '>', markersize=14,  label='FOA128', markerfacecolor='none', markeredgewidth=2.5)
    axs[0].plot(sample_results_by_encoding_LA[4, :], LA_array[3, :], 's', markersize=14, label='FOA32', markerfacecolor='none', markeredgewidth=2.5)

    axs[0].axis([0, 100, 0, 0.3])
    axs[0].tick_params(labelsize=20)
    axs[0].grid(True)
    axs[0].legend(loc='lower right', fontsize=20)
    axs[0].set_title('LA (single source)', fontsize=20, fontweight='bold')
    axs[0].set_xlabel('MUSHRA scores', fontsize=20)
    axs[0].set_ylabel('LA (NSIM)', fontsize=20)

    # Plot 2
    axs[1].plot(mushra_multi_mean[0], LA_multi[0], '*', markersize=14, label='3OA512', markerfacecolor='none', markeredgewidth=2.5)
    axs[1].plot(mushra_multi_mean[1], LA_multi[1], '^', markersize=14, label='3OA384', markerfacecolor='none', markeredgewidth=2.5)
    axs[1].plot(mushra_multi_mean[2], LA_multi[2], 'o', markersize=14, label='3OA256', markerfacecolor='none', markeredgewidth=2.5)
    axs[1].plot(mushra_multi_mean[3], LA_multi[3], 'p', markersize=14, label='FOA128', markerfacecolor='none', markeredgewidth=2.5)
    axs[1].plot(mushra_multi_mean[4], LA_multi[4], 'd', markersize=14, label='FOA96', markerfacecolor='none', markeredgewidth=2.5)
    axs[1].plot(mushra_multi_mean[5], LA_multi[5], 'h', markersize=14, label='FOA64', markerfacecolor='none', markeredgewidth=2.5)
    axs[1].plot(mushra_multi_mean[6], LA_multi[6], 'h', markersize=14, label='FOA32', markerfacecolor='none', markeredgewidth=2.5)

    axs[1].tick_params(labelsize=20)
    axs[1].grid(True)
    axs[1].legend(loc='upper left', fontsize=20)
    axs[1].set_xlabel('MUSHRA scores', fontsize=20)
    axs[1].set_ylabel('LA (NSIM)', fontsize=20)
    axs[1].set_title('LA (multiple sources)', fontsize=20, fontweight='bold')


    # Plot 3
    labels_df = results_df["srcdir"]
    labels = labels_df[0::5]

    axs[2].plot(sample_results_by_encoding_LA[1:, 0], LA_array[:, 0], '*', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[2].plot(sample_results_by_encoding_LA[1:, 1], LA_array[:, 1], 's', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[2].plot(sample_results_by_encoding_LA[1:, 2], LA_array[:, 2], 'd', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[2].plot(sample_results_by_encoding_LA[1:, 3], LA_array[:, 3], '*', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[2].plot(sample_results_by_encoding_LA[1:, 4], LA_array[:, 4], 's', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[2].plot(sample_results_by_encoding_LA[1:, 5], LA_array[:, 5], '^', markersize=14, markerfacecolor='none', markeredgewidth=2.5)

    axs[2].axis([0, 100, 0, 0.25])
    axs[2].tick_params(axis='both', which='major', labelsize=20)
    axs[2].grid(True)
    axs[2].legend(labels, loc='upper left', fontsize=18)
    axs[2].set_ylim([0, 0.32])
    axs[2].set_title('LA (single source)', fontsize=20, fontweight='bold')
    axs[2].set_xlabel('MUSHRA scores', fontsize=20)
    axs[2].set_ylabel('LA (NSIM)', fontsize=20)


    # Plot 4

    mushra_multi_mean = mushra_multi_df.iloc[:, 3:].mean(axis=1, skipna=True).tolist()
    mushra_multi_mean = np.reshape(mushra_multi_mean, (6, 8)).T[1:, :].flatten(order='F')
    label = mushra_multi_df["label"]
    label = label[0::8]

    axs[3].plot(mushra_multi_mean[0:7], LA_multi_values[0:7], '*', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[3].plot(mushra_multi_mean[7:14], LA_multi_values[7:14], '^', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[3].plot(mushra_multi_mean[14:21], LA_multi_values[14:21], 'o', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[3].plot(mushra_multi_mean[21:28], LA_multi_values[21:28], 'p', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[3].plot(mushra_multi_mean[28:35], LA_multi_values[28:35], 'd', markersize=14, markerfacecolor='none', markeredgewidth=2.5)
    axs[3].plot(mushra_multi_mean[35:42], LA_multi_values[35:42], 'h', markersize=14, markerfacecolor='none', markeredgewidth=2.5)

    axs[3].tick_params(labelsize=20)
    axs[3].grid(True)
    axs[3].legend(label, loc='upper left', fontsize=20)
    axs[3].set_xlabel('MUSHRA scores', fontsize=20)
    axs[3].set_ylabel('LA(NSIM)', fontsize=20)
    axs[3].set_title('LA (multiple sources)', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Save the figure as a PDF
    fig.savefig("LA.pdf", bbox_inches='tight')


if __name__ == '__main__':

    ref_files_exp1 = ['castanets_fixed_A60_E60_HOA_REF.wav',
                    'glock_fixed_A60_E60_HOA_REF.wav',
                    'castanetsRev_dynamic_A0_A360_E30_HOA_REF.wav',
                    'burstyPinkRev_dynamic_A0_A360_E30_HOA_REF.wav',
                    'castanetsRev_dynamic_A60_E0_E180_HOA_REF.wav',
                    'burstyPinkRev_dynamic_A60_E0_E180_HOA_REF.wav'
                    ]

    deg_files_exp1 = [
        'castanets_fixed_A60_E60_HOA_512k.wav',
        'castanets_fixed_A60_E60_HOA_256k.wav',
        'castanets_fixed_A60_E60_FOA_128k.wav',
        'castanets_fixed_A60_E60_FOA_32k.wav',

        'glock_fixed_A60_E60_HOA_512k.wav',
        'glock_fixed_A60_E60_HOA_256k.wav',
        'glock_fixed_A60_E60_FOA_128k.wav',
        'glock_fixed_A60_E60_FOA_32k.wav',

        'castanetsRev_dynamic_A0_A360_E30_HOA_512k.wav',
        'castanetsRev_dynamic_A0_A360_E30_HOA_256k.wav',
        'castanetsRev_dynamic_A0_A360_E30_FOA_128k.wav',
        'castanetsRev_dynamic_A0_A360_E30_FOA_32k.wav',

        'burstyPinkRev_dynamic_A0_A360_E30_HOA_512k.wav',
        'burstyPinkRev_dynamic_A0_A360_E30_HOA_256k.wav',
        'burstyPinkRev_dynamic_A0_A360_E30_FOA_128k.wav',
        'burstyPinkRev_dynamic_A0_A360_E30_FOA_32k.wav',

        'castanetsRev_dynamic_A60_E0_E180_HOA_512k.wav',
        'castanetsRev_dynamic_A60_E0_E180_HOA_256k.wav',
        'castanetsRev_dynamic_A60_E0_E180_FOA_128k.wav',
        'castanetsRev_dynamic_A60_E0_E180_FOA_32k.wav',

        'burstyPinkRev_dynamic_A60_E0_E180_HOA_512k.wav',
        'burstyPinkRev_dynamic_A60_E0_E180_HOA_256k.wav',
        'burstyPinkRev_dynamic_A60_E0_E180_FOA_128k.wav',
        'burstyPinkRev_dynamic_A60_E0_E180_FOA_32k.wav'
    ]


    ref_files_exp2 = [
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_HOA_REF.wav',
        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_HOA_REF.wav',
        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_HOA_REF.wav',
        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_HOA_REF.wav',
        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_HOA_REF.wav',
        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_HOA_REF.wav'
    ]

    deg_files_exp2 = [
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_HOA_512k.wav',
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_HOA_384k.wav',
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_HOA_256k.wav',
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_FOA_128k.wav',
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_FOA_96k.wav',
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_FOA_64k.wav',
        'castanetsorgRev_dynamic_A-90_A90_E60_60_glockshortRev_fixed_A60_E15_FOA_32k.wav',

        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_HOA_512k.wav',
        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_HOA_384k.wav',
        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_HOA_256k.wav',
        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_FOA_128k.wav',
        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_FOA_96k.wav',
        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_FOA_64k.wav',
        'tabbels_dynamic_A-90_A90_E-45_-45_xylophone_fixed_A-90_E45_FOA_32k.wav',

        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_HOA_512k.wav',
        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_HOA_384k.wav',
        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_HOA_256k.wav',
        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_FOA_128k.wav',
        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_FOA_96k.wav',
        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_FOA_64k.wav',
        'bPinkRev_dynamic_A90_E0_E180_castanets_fixed_A90_E-15_FOA_32k.wav',

        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_HOA_512k.wav',
        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_HOA_384k.wav',
        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_HOA_256k.wav',
        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_FOA_128k.wav',
        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_FOA_96k.wav',
        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_FOA_64k.wav',
        'female_dynamic_A-90_A90_E30_30_babble_fixed_A90_E0_FOA_32k.wav',

        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_HOA_512k.wav',
        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_HOA_384k.wav',
        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_HOA_256k.wav',
        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_FOA_128k.wav',
        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_FOA_96k.wav',
        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_FOA_64k.wav',
        'trianglesst_dynamic_A60_E0_E180_trianglesroll_fixed_A60_E0_FOA_32k.wav',

        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_HOA_512k.wav',
        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_HOA_384k.wav',
        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_HOA_256k.wav',
        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_FOA_128k.wav',
        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_FOA_96k.wav',
        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_FOA_64k.wav',
        'xylophone_dynamic_A60_E0_E180_piano_fixed_A60_E0_FOA_32k.wav'
    ]

    intensity_threshold = -180
    elc = 0
    ignorefreqbands = 0
    path = "audiofiles/"

    # experiment = 1
    # validate_ambiqual(ref_files_exp1, deg_files_exp1, path, intensity_threshold, elc, ignorefreqbands, experiment)
    #
    # experiment = 2
    # validate_ambiqual(ref_files_exp2, deg_files_exp2, path, intensity_threshold, elc, ignorefreqbands, experiment)

    plot_lq()
    plot_la()