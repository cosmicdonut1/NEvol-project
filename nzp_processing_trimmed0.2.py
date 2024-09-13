import os
import numpy as np
import pandas as pd
import mne
from utils import load_buffers, load_task_markers
from matplotlib import pyplot as plt

# Установка log level для вывода информации о фильтрации
mne.set_log_level('info')


# Mirror padding function
def mirror_padding(data, pad_len):
    return np.concatenate([np.flip(data[:, :pad_len], axis=1),
                           data,
                           np.flip(data[:, -pad_len:], axis=1)], axis=1)


# Function for filtering EEG data
def filter_eeg_data(df, sfreq, low_freq, high_freq, pad_len=256):
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    raw_data = df[channel_names].to_numpy().T
    timestamps = df['timestamps'].to_numpy()
    data_padded = mirror_padding(raw_data, pad_len)
    raw = mne.io.RawArray(data_padded, info)
    raw.filter(low_freq, high_freq, method='iir')
    raw.notch_filter(freqs=50.0, method='iir')
    filtered_data = raw.get_data()[:, pad_len:-pad_len]
    timestamps = timestamps[:filtered_data.shape[1]]
    df_filtered = pd.DataFrame(data=filtered_data.T, columns=channel_names)
    df_filtered['timestamps'] = timestamps
    return df_filtered


# Function to trim EEG data
def trim_eeg_data(df, trim_duration=0.2):
    timestamps = df['timestamps'].values
    start_time = timestamps[0] + trim_duration
    end_time = timestamps[-1] - trim_duration
    start_idx = np.searchsorted(timestamps, start_time)  # Индекс начала
    end_idx = np.searchsorted(timestamps, end_time)  # Индекс конца
    trimmed_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    return trimmed_df


# Function to plot EEG data
def plot_eeg_data(df, title):
    plt.figure(figsize=(10, 6))
    for channel in channel_names:
        plt.plot(df['timestamps'], df[channel], label=channel)
    plt.xlabel('Timestamps')
    plt.ylabel('Amplitude (μV)')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


# Function to visualize topomap
def plot_topomap(data, title, sfreq=250):
    info = mne.create_info(ch_names=channel_names_subset, sfreq=sfreq, ch_types='eeg')
    evoked = mne.EvokedArray(data, info)
    evoked.set_montage(montage)
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    mne.viz.plot_topomap(evoked.data.mean(axis=1), evoked.info, axes=ax, show=True, cmap='RdBu_r')
    ax.set_title(title, fontsize=10)
    plt.show()


# Function to process a single file with optional visualization
def process_file(task_marker_path, signal_data_path, visualize=True):
    df_tasks = load_task_markers(task_marker_path)
    df_buffers = load_buffers(signal_data_path)
    df_buffers = df_buffers.sort_values('timestamps').reset_index(drop=True)
    df_tasks = df_tasks.sort_values('timestamps').reset_index(drop=True)

    global channel_names
    channel_names = ['FP1', 'FP2', 'C3', 'C4', 'FC1', 'FC2', 'CP1', 'CP2']
    relevant_channels = 8
    df_buffers_raw = df_buffers.iloc[:, :relevant_channels]
    df_buffers_raw.columns = channel_names
    df_buffers_raw['timestamps'] = df_buffers['timestamps']

    df_markers = df_tasks[df_tasks['event_ids'].isin([1, 2, 3, 4, 99, 100])]
    markers_map = {1: 'left_start', 2: 'left_end', 3: 'right_start',
                   4: 'right_end', 99: 'rest_start', 100: 'rest_end'}

    onsets, durations, descriptions = [], [], []
    start_times = {}

    for index, row in df_markers.iterrows():
        event_id = row['event_ids']
        timestamp = row['timestamps']
        if event_id in markers_map:
            description = markers_map[event_id]
            if 'start' in description:
                start_times[description.split('_')[0]] = timestamp
            elif 'end' in description:
                event_type = description.split('_')[0]
                if event_type in start_times:
                    start_time = start_times[event_type]
                    duration = timestamp - start_time
                    onsets.append(start_time)
                    durations.append(duration)
                    descriptions.append(event_type.upper())
                    del start_times[event_type]

    left_events, right_events, rest_events = [], [], []
    for onset, duration, description in zip(onsets, durations, descriptions):
        event_df = df_buffers_raw[(df_buffers_raw['timestamps'] >= onset) &
                                  (df_buffers_raw['timestamps'] <= (onset + duration))]
        if description == 'LEFT':
            left_events.append(event_df)
        elif description == 'RIGHT':
            right_events.append(event_df)
        elif description == 'REST':
            rest_events.append(event_df)

    sfreq = 250
    low_freq = 8.0
    high_freq = 30.0

    # Filter events
    left_events_filtered = [filter_eeg_data(df, sfreq, low_freq, high_freq) for df in left_events]
    right_events_filtered = [filter_eeg_data(df, sfreq, low_freq, high_freq) for df in right_events]
    rest_events_filtered = [filter_eeg_data(df, sfreq, low_freq, high_freq) for df in rest_events]

    # Trim filtered events
    left_events_filtered_trimmed = [trim_eeg_data(df) for df in left_events_filtered]
    right_events_filtered_trimmed = [trim_eeg_data(df) for df in right_events_filtered]

    # Обновим оригинальные списки
    left_events_filtered = left_events_filtered_trimmed
    right_events_filtered = right_events_filtered_trimmed

    # Визуализация, если включено
    if visualize:
        for i, df in enumerate(left_events_filtered[:3]):
            plot_eeg_data(df, title=f"Trimmed Left Event {i + 1}")
        for i, df in enumerate(right_events_filtered[:3]):
            plot_eeg_data(df, title=f"Trimmed Right Event {i + 1}")

        # Визуализируем топомапы
        global channel_names_subset
        channel_names_subset = ['C3', 'C4', 'FC1', 'FC2', 'CP1', 'CP2']
        global montage
        montage = mne.channels.make_standard_montage('standard_1020')

        num_events = min(len(left_events_filtered), len(right_events_filtered))
        fig, axes = plt.subplots(num_events, 2, figsize=(12, num_events * 2))

        for i in range(num_events):
            # Left events
            data_left = left_events_filtered[i][channel_names_subset].to_numpy().T
            info_left = mne.create_info(ch_names=channel_names_subset, sfreq=250, ch_types='eeg')
            evoked_left = mne.EvokedArray(data_left, info_left)
            evoked_left.set_montage(montage)
            mne.viz.plot_topomap(evoked_left.data.mean(axis=1), evoked_left.info, axes=axes[i, 0], show=False,
                                 cmap='RdBu_r')
            axes[i, 0].set_title(f'Left Hand Activation {i + 1}', fontsize=10)

            # Right events
            data_right = right_events_filtered[i][channel_names_subset].to_numpy().T
            info_right = mne.create_info(ch_names=channel_names_subset, sfreq=250, ch_types='eeg')
            evoked_right = mne.EvokedArray(data_right, info_right)
            evoked_right.set_montage(montage)
            mne.viz.plot_topomap(evoked_right.data.mean(axis=1), evoked_right.info, axes=axes[i, 1], show=False,
                                 cmap='RdBu_r')
            axes[i, 1].set_title(f'Right Hand Activation {i + 1}', fontsize=10)

        plt.tight_layout()
        plt.show()

    return left_events_filtered, right_events_filtered, rest_events_filtered


# Main function to process all files in the directory
def main():
    base_dir = 'processed_data/motor_imagery'
    output_dir = 'processed_data/filtered_events/'
    os.makedirs(output_dir, exist_ok=True)

    visualize = False  # Установите False, чтобы отключить визуализацию
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            task_marker_path = os.path.join(base_dir, d, 'task_data/task_markers.npz')
            signal_data_path = os.path.join(base_dir, d, 'signal_data/')
            if os.path.exists(task_marker_path) and os.path.exists(signal_data_path):
                left_events_filtered, right_events_filtered, rest_events_filtered = process_file(task_marker_path,
                                                                                                 signal_data_path,
                                                                                                 visualize)

                # Сохранение отфильтрованных событий в CSV
                for i, df in enumerate(left_events_filtered):
                    df.drop(columns=['FP1', 'FP2'], inplace=True, errors='ignore')
                    df.to_csv(os.path.join(output_dir, f'{d}_filtered_left_event_{i + 1}.csv'), index=False)

                for i, df in enumerate(right_events_filtered):
                    df.drop(columns=['FP1', 'FP2'], inplace=True, errors='ignore')
                    df.to_csv(os.path.join(output_dir, f'{d}_filtered_right_event_{i + 1}.csv'), index=False)

                for i, df in enumerate(rest_events_filtered):
                    df.drop(columns=['FP1', 'FP2'], inplace=True, errors='ignore')
                    df.to_csv(os.path.join(output_dir, f'{d}_filtered_rest_event_{i + 1}.csv'), index=False)

    print("Filtered events have been saved successfully for all files.")


# Вызов функции main
if __name__ == '__main__':
    main()