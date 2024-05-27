from datetime import datetime, timezone
import numpy as np
import pandas as pd
import os

from pylsl import local_clock
import time

def read_npz_file(filename):
    """
    Reads an .npz file and prints its contents.

    Args:
        filename (str): The path to the .npz file.

    Returns:
        dict: A dictionary containing the arrays stored in the .npz file.
    """
    with np.load(filename) as data:
        # Iterate over each array in the .npz file and print its contents
        for key, value in data.items():
            print(f"Key: {key}")
            print(f"Value:\n{value}\n")
        return dict(data)


def lsl_to_datetime(lsl_timestamp):
    """
    Convert LSL timestamp to NumPy datetime64 timestamp.

    Parameters:
    lsl_timestamp (float): The LSL timestamp to convert.

    Returns:
    np.datetime64: The corresponding NumPy datetime64 timestamp.
    """
    # Convert LSL timestamp to Python datetime
    dt_timestamp = datetime.utcfromtimestamp(lsl_timestamp)
    # Convert Python datetime to NumPy datetime64
    np_timestamp = np.datetime64(dt_timestamp)
    return np_timestamp


def load_task_markers(file_path):
    data = np.load(file_path)

    # Display the names of the arrays stored in the .npz file
    print("Array names in the .npz file:", data.files)  # signal_thread.join()

    # Create the DataFrame
    df_task = pd.DataFrame({
        'event_ids': data['event_ids'][:, 0],
        'timestamps': data['timestamps']
    })

    return df_task


# Function to process a single .npz file
def process_npz_file(file_path):
    data = np.load(file_path)
    timestamps = data['timestamps']
    buffer = data['buffer']

    # Ensure buffer is 2D
    if buffer.ndim == 1:
        buffer = buffer.reshape(-1, 1)

    # Combine buffer and timestamps
    combined = np.hstack((buffer, timestamps.reshape(-1, 1)))

    # Create DataFrame
    df = pd.DataFrame(combined, columns=[f'buffer_col_{i}' for i in range(buffer.shape[1])] + ['timestamps'])

    return df


def load_buffers(folder_path):
    dfs = []
    index = 1

    # Iterate through all .npz files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npz'):
            file_path = os.path.join(folder_path, file_name)
            df = process_npz_file(file_path)
            df['epoch_number'] = index
            index += 1
            dfs.append(df)

    # Concatenate all DataFrames
    df_buffers = pd.concat(dfs, ignore_index=True)

    return df_buffers



