# import sys
# import os
# import time

# import matplotlib as plt

from threading import Thread, Event

from buffer import Buffer
from lsl_manager import *

import numpy as np
import pandas as pd

# from task import execute_task
from utils import load_task_markers, load_buffers
from mne import create_info
from mne.epochs import EpochsArray

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer

if __name__ == "__main__":
    # Initialize the variables
    info = {'start_time': time.time()}

    mode = "train"
    # mode = "predict"

    epoch_duration = 2
    sampling_rate = 125
    channel_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    ls_markers = []
    eeg_signals = np.zeros((len(channel_names), epoch_duration * sampling_rate))

    buffer = Buffer(duration=epoch_duration, sampling_rate=sampling_rate, num_channels=17)
    stop_event = Event()

    bool_s_stream_status = check_stream("UN-2023.04.61")
    bool_m_stream_status = check_stream("task_stream")

    if bool_s_stream_status and bool_m_stream_status:
        signal_thread = Thread(target=read_signal_stream, args=("UN-2023.04.61", buffer, stop_event))
        signal_thread.start()

        # task_initiation_thread = Thread(target=execute_task, args=())
        # task_initiation_thread.start()

        task_thread = Thread(target=read_task_stream, args=("task_stream", ls_markers))
        task_thread.start()
        task_thread.join()
        stop_event.set()

    info['end_time'] = time.time()

    # Wait for task thread to finish
    print("Task Ended. Now preprocessing...")

    if mode == "train":

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
        mask = df_buffers.set_index(['epoch_number', 'phase']).index.isin(filtered_groups.set_index(['epoch_number', 'phase']).index)
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

    elif mode == "predict":
        # load the pickle file
        # use the buffer and the loaded model from 1 to predict the user state
        # utilize the user state to call required functions
        pass