import time
import numpy as np
import pandas as pd
import mne
import time

import matplotlib.pyplot as plt
# from mne.time_frequency import psd_multitaper
from datetime import datetime

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

channel_mapping = {
    0: 'Fp1', 1: 'Fp2', 2: 'C3', 3: 'C4', 4: 'T5', 5: 'T6', 6: 'O1', 7: 'O2',
    8: 'F7', 9: 'F8', 10: 'F3', 11: 'F4', 12: 'T3', 13: 'T4', 14: 'P3', 15: 'P4'
}

parameters_file = r'D:\Code\NEvol_git\Sandbox\Interaction_with_visualisation\parameters.txt'
is_moving = False
is_rotating_left = False
is_rotating_right = False

def create_power_table(spectrum, band_name, channel_names):
    ls_spectrum = spectrum.get_data().reshape(4, 8)
    df = pd.DataFrame(ls_spectrum, columns=[f'Channel_{i}' for i in range(1, 17)])
    df.columns = channel_names
    print(band_name, "\n")
    print("Average Fz: ", df['Fp1'].mean(),"\nAverage Cz: ", df['C3'].mean(),"\nAverage Pz :", df['P3'].mean())
    print("--------------------------------------------------------------------------")


def analyze_signal(mode, df_buffer, channel_names):
    if mode == "bandpower":
        # Clear plot
        plt.clf()
        plt.close('all')

        channel_data = df_buffer.iloc[:, :-1].values.T
        timestamps = pd.to_datetime(df_buffer['timestamp'])

        info = mne.create_info(
            ch_names=channel_names,
            sfreq=125.0,  # Assuming the data is sampled at 1 Hz; adjust as necessary
            ch_types=['eeg'] * 16
        )

        # Create Raw object
        raw = mne.io.RawArray(channel_data, info)
        raw.set_montage('standard_1005')

        # raw.compute_psd()

        # Bandpass filter for theta and alpha bands
        theta_band = (4, 7)
        alpha_band = (8, 13)

        # Epoching the data
        epochs = mne.make_fixed_length_epochs(raw, duration=0.5, overlap=0.0, preload=True)

        # Compute PSD for epochs
        # psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=2, fmax=30, n_fft=256)

        theta_spectrum = epochs.compute_psd(method='multitaper', fmin=theta_band[0], fmax=theta_band[1], tmin=0, tmax=2)
        # theta_spectrum.plot()

        alpha_spectrum = epochs.compute_psd(method='multitaper', fmin=theta_band[0], fmax=theta_band[1], tmin=0, tmax=2)
        # alpha_spectrum.plot()

        # Define the electrode names you want to include in the table
        # electrode_names = ['Fz', 'Pz', 'Cz']

        # Create tables for theta and alpha bands
        create_power_table(theta_spectrum, 'theta', channel_names)
        create_power_table(alpha_spectrum, 'alpha', channel_names)

        # theta_spectrum, alpha_spectrum are of the shape (4,8,1) (epochs, electrodes, _)
        # write a for loop on theta and derived a table with columns (Epoch_number, band = theta, Fz, Pz and Cz), the table should contain values from theta_spectrum

        # -------------

def write_parameters(is_moving, is_rotating_left, is_rotating_right):
    with open(parameters_file, 'w') as f:
        f.write(f"{is_moving},{is_rotating_left},{is_rotating_right}")

def classify_eyeblinks(mode, df_buffer, channel_names):
    global is_moving
    global is_rotating_left
    global is_rotating_right
    channel_data = df_buffer.iloc[:, :-1].values.T
    timestamps = pd.to_datetime(df_buffer['timestamp'])

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=125.0,  # Assuming the data is sampled at 1 Hz; adjust as necessary

        # change this for different electrode configuration
        ch_types=['eeg'] * 16
    )

    # Create Raw object
    raw = mne.io.RawArray(channel_data, info)
    raw.set_montage('standard_1005')
    # raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

    # Create EOG channel
    # Which channels must be selected for EOG detection
    # eog_channel = df_buffer.iloc[:, 0].values * 1e-6  # Convert µV to V
    eog_channel = df_buffer['Fp1'].values * 1e-6  # Convert µV to V
    eog_info = mne.create_info(['EOG'], sfreq=info['sfreq'], ch_types=['eog'])
    eog_raw = mne.io.RawArray(eog_channel[None, :], eog_info)

    # Merge EEG and EOG data before filtering
    raw_combined = raw.add_channels([eog_raw])

    raw_combined.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

    eog_epochs = mne.preprocessing.create_eog_epochs(raw_combined, ch_name='EOG')  # Specify the new EOG channel
    eog_events = eog_epochs.events

    print(eog_events)

    ### --- SENDING THE COMMAND TO CONTROL VISUALISATION --- ###

    control_parameter = len(eog_events)
    print(control_parameter)

    if control_parameter == 1:
        is_moving = not is_moving
        is_rotating_left = False
        is_rotating_right = False
    elif control_parameter == 2:
        is_rotating_right = True
        is_rotating_left = False
    elif control_parameter == 3:
        is_rotating_left = True
        is_rotating_left = False

    write_parameters(is_moving, is_rotating_left, is_rotating_right)


    ### --- END OF SENDING THE COMMAND TO CONTROL VISUALISATION --- ###


    eog_events_df = pd.DataFrame(eog_events, columns=['sample', 'prev_event_id', 'event_id'])

    # Step 4: Visualize raw and filtered data with detected blinks
    fig, axs = plt.subplots(len(df_buffer.columns), 2, figsize=(15, 3 * len(df_buffer.columns)))

    for i, channel in enumerate(df_buffer.columns):
        # Raw signal display
        axs[i, 0].plot(df_buffer.index / info['sfreq'], df_buffer[channel], label='Raw Data')
        axs[i, 0].set_title(f'Raw Data (Channel {i})')
        axs[i, 0].set_ylabel('Amplitude (µV)')
        if i == len(df_buffer.columns) - 1:
            axs[i, 0].set_xlabel('Time (s)')

        # Filtered signal display
        axs[i, 1].plot(raw_combined.times, raw_combined.get_data(picks=[i])[0] * 1e6,
                       label='Filtered Data')  # Convert back to µV

        # Mark detected blinks on the filtered data
        blink_samples = eog_events_df['sample'].values
        blink_times = blink_samples / info['sfreq']
        for blink_time in blink_times:
            axs[i, 1].axvline(x=blink_time, color='r', linestyle='--',
                              label='Detected Blink' if blink_time == blink_times[0] else "")

        axs[i, 1].set_title(f'Filtered Data (Channel {i})')
        axs[i, 1].set_ylabel('Amplitude (µV)')
        if i == len(df_buffer.columns) - 1:
            axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].legend()

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Display the plot with detected events
    #plt.show()


        # ------------------------------------------------------------------------------
        # Extract labels for classification (this should be provided)
        # Generate random labels for classification
        # np.random.seed(42)  # For reproducibility
        # labels = np.random.randint(0, 2, size=len(epochs))  # Assuming binary classification (workload levels)

        # # Feature extraction using CSP
        # csp = CSP(n_components=3, reg=None, log=True, cov_est='epoch')
        #
        # # LDA classifier
        # lda = LinearDiscriminantAnalysis()
        #
        # # Pipeline
        # pipe = Pipeline([('CSP', csp), ('LDA', lda)])
        #
        # # Cross-validation
        # cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        # scores = cross_val_score(pipe, epochs.get_data(), labels, cv=cv, n_jobs=1)
        #
        # print("Cross-validation accuracy: %f ± %f" % (scores.mean(), scores.std()))
        #
        # # Train the classifier on the whole dataset
        # pipe.fit(epochs.get_data(), labels)
        #
        # # Apply the classifier to new data (1-second epochs from different tasks)
        # # Assuming `new_epochs` contains the new data
        # new_data = epochs.get_data()
        # predictions = pipe.predict(new_data)
        #
        # # Statistical comparison using permutation tests
        # # Assuming `condition_1` and `condition_2` are arrays of classifier outputs for different workload conditions
        # # For simplicity, splitting the predictions into two arbitrary conditions
        # condition_1 = predictions[:len(predictions) // 2]
        # condition_2 = predictions[len(predictions) // 2:]

            # ------------------------------------------------

        # raw.filter(0, 10, fir_design='firwin')
        # events = np.array([[i, 0, 1] for i in range(len(timestamps))])
        # picks = mne.pick_types(raw.info, meg="grad", eeg=True)

        # Construct Epochs
        # event_id, tmin, tmax = 1, -1.0, 2.0
        # baseline = (None, 0)
        # epochs = mne.Epochs(
        #     raw,
        #     events,
        #     event_id,
        #     tmin,
        #     tmax,
        #     baseline=baseline,
        #     preload=True,
        # )
        # epochs.compute_psd().plot_topomap(ch_type="grad", normalize=False, contours=0)

