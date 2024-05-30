import time
import numpy as np
import pandas as pd
import mne

import matplotlib.pyplot as plt
from utils import load_task_markers, load_buffers
# from mne.time_frequency import psd_multitaper
from datetime import datetime

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from utils import load_task_markers, load_buffers
from mne import create_info
from mne.epochs import EpochsArray


from mne.decoding import Vectorizer

channel_mapping = {0: 'Fz', 1: 'C3', 2: 'Cz', 3: 'C4', 4: 'Pz', 5: 'PO7', 6: 'Oz', 7: 'PO8'}


def create_power_table(spectrum, band_name, channel_names):
    ls_spectrum = spectrum.get_data().reshape(4, 8)
    df = pd.DataFrame(ls_spectrum, columns=[f'Channel_{i}' for i in range(1, 9)])
    df.columns = channel_names
    print(band_name, "\n")
    print("Average Fz: ", df['Fz'].mean(),"\nAverage Cz: ", df['Cz'].mean(),"\nAverage Pz :", df['Pz'].mean())
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
            ch_types=['eeg'] * 8
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


def process_imagery_data(channel_names):
    # load the task and signal files stored in the disk
    # Based on the task timestamps, subset the signals and annotate the events
    # create epoch data that includes the signal and events
    # use the epoch data to train a classifier
    # save the classifier model as a pickle file for prediction later

    # Load the .npz file for task
    file_path = 'processed_data/task_data/task_markers.npz'  # Replace with your file path
    df_tasks = load_task_markers(file_path)

    # Load the .npz file for buffer
    folder_path = 'processed_data/signal_data/'
    df_buffers = load_buffers(folder_path)

    print(df_buffers)
    print("Task Ended. Now preprocessing...")

    df_buffers = df_buffers.sort_values('timestamps').reset_index(drop=True)
    df_tasks = df_tasks.sort_values('timestamps').reset_index(drop=True)

    # Perform an as of merge to find the closest earlier and later event_id
    df_buffers['prev_event_id'] = pd.merge_asof(df_buffers, df_tasks,
                                                left_on='timestamps', right_on='timestamps',
                                                direction='backward')['event_ids']

    df_buffers['next_event_id'] = pd.merge_asof(df_buffers, df_tasks,
                                                left_on='timestamps', right_on='timestamps',
                                                direction='forward')['event_ids']

    markers = {
        'forward_start': [1],
        'forward_end': [2],
        'reverse_start': [3],
        'reverse_end': [4],
        'rest_start': [99],
        'rest_end': [100],
        'task_start': [-1],
        'task_end': [-2]
    }

    def determine_phase(row):
        if row['prev_event_id'] in markers['forward_start'] and row['next_event_id'] in markers['forward_end']:
            return 'forward'
        elif row['prev_event_id'] in markers['reverse_start'] and row['next_event_id'] in markers['reverse_end']:
            return 'reverse'
        elif row['prev_event_id'] in markers['rest_start'] and row['next_event_id'] in markers['rest_end']:
            return 'rest'
        else:
            return 'unknown'

    df_buffers['phase'] = df_buffers.apply(determine_phase, axis=1)

    grouped = df_buffers.groupby(['epoch_number', 'phase']).size().reset_index(name='count')

    valid_phases = ['forward', 'reverse', 'rest']
    filtered_groups = grouped[(grouped['phase'].isin(valid_phases)) & (grouped['count'] == 250)]
    mask = df_buffers.set_index(['epoch_number', 'phase']).index.isin(
        filtered_groups.set_index(['epoch_number', 'phase']).index)
    filtered_df_buffers = df_buffers[mask].reset_index(drop=True)

    columns_to_select = df_buffers.columns[:8].tolist() + ['epoch_number', 'timestamps', 'phase']
    final_df_buffers = filtered_df_buffers[columns_to_select]

    rename_dict = {old_name: new_name for old_name, new_name in zip(final_df_buffers.columns[:8], channel_names)}
    final_df_buffers = final_df_buffers.rename(columns=rename_dict)

    final_df_buffers.groupby(['epoch_number', 'phase']).size().reset_index(name='count')

    # Creating epochs data
    epochs_data = []
    events = []

    # Need to add rest:3, I have remove it temporarily as when I record for test I am not getting pure rest (250 s) signals.
    # but if we run the psychopy iterations, we will have representation of rest state

    event_id = {'forward': 1, 'reverse': 2}
    phases = ['forward', 'reverse', 'rest']

    for epoch_number, phase in final_df_buffers.groupby(['epoch_number', 'phase']):
        phase_name = epoch_number[1]
        if phase_name not in phases:
            continue
        epoch_data = phase[channel_names].values.T  # Transpose to get (n_channels, n_times)
        epochs_data.append(epoch_data)
        events.append([len(epochs_data) - 1, 0, event_id[phase_name]])

    epochs_data = np.array(epochs_data)  # Shape should be (n_epochs, n_channels, n_times)
    events = np.array(events)  # Shape should be (n_epochs, 3)

    sfreq = 125

    # Create MNE info structure
    info = create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')

    # Create MNE Epochs object
    epochs = EpochsArray(epochs_data, info, events, event_id=event_id, tmin=0)

    X = epochs.get_data(copy=False)  # Shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2]  # Shape: (n_epochs,)

    # Reshape the data to (n_samples, n_features)
    X = X.reshape(len(X), -1)

    # Create and train a classifier
    clf = make_pipeline(Vectorizer(), RandomForestClassifier(n_estimators=100))
    clf.fit(X, y)

    test_index = 0  # Use the first training data point for prediction
    test_sample = X[test_index].reshape(1, -1)  # Reshape to (1, n_features)
    predicted_label = clf.predict(test_sample)
    actual_label = y[test_index]
    print(f"Predicted label: {predicted_label[0]}, Actual label: {actual_label}")


def classify_eyeblinks(df_buffer, channel_names):
    channel_data = df_buffer.iloc[:, :-1].values.T
    timestamps = pd.to_datetime(df_buffer['timestamp'])

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=125.0,  # Assuming the data is sampled at 1 Hz; adjust as necessary

        # change this for different electrode configuration
        ch_types=['eeg'] * 8
    )

    # Create Raw object
    raw = mne.io.RawArray(channel_data, info)
    raw.set_montage('standard_1005')
    # raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

    # Create EOG channel
    # Which channels must be selected for EOG detection
    # eog_channel = df_buffer.iloc[:, 0].values * 1e-6  # Convert µV to V
    eog_channel = df_buffer['Fz'].values * 1e-6  # Convert µV to V
    eog_info = mne.create_info(['EOG'], sfreq=info['sfreq'], ch_types=['eog'])
    eog_raw = mne.io.RawArray(eog_channel[None, :], eog_info)

    # Merge EEG and EOG data before filtering
    raw_combined = raw.add_channels([eog_raw])

    raw_combined.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

    eog_epochs = mne.preprocessing.create_eog_epochs(raw_combined, ch_name='EOG')  # Specify the new EOG channel
    eog_events = eog_epochs.events

    print(eog_events)
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
    plt.show()


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

