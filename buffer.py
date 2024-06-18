import numpy as np
import pandas as pd
import config
import time
import os
from datetime import datetime
from bufferManager import BufferManager


class Buffer:
    def __init__(self, duration, sampling_rate, num_channels, save_path, buffer_manager):
        self.electrodes = config.device_details['relevant_channels_from_device']
        self.duration = duration
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.buffer_size = int(duration * sampling_rate)
        self.buffer = np.zeros((self.buffer_size, num_channels))
        self.markers = np.array([])
        self.timestamps = np.zeros(self.buffer_size)
        self.current_size = 0  # Track the current number of samples in the buffer
        self.save_path = save_path
        self.buffers = buffer_manager

        # Ensure the save path exists
        os.makedirs(self.save_path, exist_ok=True)

    # def __repr__(self):
    #     return "Buffer Object with size : {}".format(self.buffer_size)

    def add_sample(self, sample, timestamp):
        if self.current_size < self.buffer_size:
            self.buffer[self.current_size] = sample
            self.timestamps[self.current_size] = timestamp
            self.current_size += 1
        else:
            # Save and clear the buffer when full
            current_time = time.time()
            filename = os.path.join(self.save_path, f"buffer_{int(current_time)}.npz")

            self.buffers.add_buffer(self)
            self.save_buffer_to_disk(filename)
            self.clear_buffer()

            # Add the new sample after clearing
            self.buffer[0] = sample
            self.timestamps[0] = timestamp
            self.current_size = 1

    def save_buffer_to_disk(self, filename):
        np.savez(filename, buffer=self.buffer, timestamps=self.timestamps)
        # self.print_buffer_shape()
        print(f"Session Buffer saved to disk! Duration = ", self.timestamps[0], " to ", self.timestamps[-1])

    def clear_buffer(self):
        print("Clearing buffer...")
        self.buffer = np.zeros((self.buffer_size, self.num_channels))
        self.timestamps = np.zeros(self.buffer_size)
        self.current_size = 0

    def get_buffer_data(self):
        return self.buffer

    def get_plottable_data(self, channel_names):
        buffer_timestamps = self.get_buffer_timestamps()

        # We are selecting the first 8 columns
        ls_select_buffer = self.buffer[:, :self.electrodes]
        selected_columns = np.column_stack((ls_select_buffer, buffer_timestamps))

        columns = channel_names + ["timestamp"]
        df_buffer = pd.DataFrame(selected_columns, columns=columns)

        return df_buffer

    def get_buffer_timestamps(self):
        return self.timestamps

    def print_buffer(self):
        print("Buffer contents:")
        print(self.buffer)

    def print_buffer_shape(self):
        print("Buffer shape: ", self.buffer.shape)
