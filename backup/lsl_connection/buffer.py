import numpy as np

class Buffer:
    def __init__(self, duration, sampling_rate, num_channels):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.buffer_size = int(duration * sampling_rate)
        self.buffer = np.zeros((self.buffer_size, num_channels))
        self.markers = np.array([])
        self.timestamps = np.zeros(self.buffer_size)

    def add_sample(self, sample, timestamp):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = sample

        self.timestamps = np.roll(self.timestamps, -1, axis=0)
        self.timestamps[-1] = timestamp

    def save_buffer(self, filename):
        np.savez(filename, buffer=self.buffer, timestamps=self.timestamps)
        print(f"Session Buffer saved to disk!")

    def clear_buffer(self):
        print("Clearing buffer...")
        self.buffer = np.zeros((self.buffer_size, 17))

    def get_buffer_data(self):
        return self.buffer

    def print_buffer(self):
        print("Buffer contents:")
        print(self.buffer)

    def print_buffer_shape(self):
        print("Buffer shape:")
        print(self.buffer.shape)
