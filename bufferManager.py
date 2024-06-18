import matplotlib.pyplot as plt
import pandas as pd

import config


class BufferManager:
    def __init__(self):
        self.buffers = []

    def add_buffer(self, buffer):
        self.buffers.append(buffer)

    def get_data(self, mode):
        df = pd.DataFrame()
        if mode == 'last':
            data = self.buffers[-1].get_plottable_data(channel_names=config.device_details['channels'])
            df = pd.concat([df, data], ignore_index=True)

        if mode == "all":
            for idx, buffer in enumerate(self.buffers):
                data = buffer.get_plottable_data(channel_names=config.device_details['channels'])
                df = pd.concat([df, data], ignore_index=True)

        return df

    def get_buffers_len(self):
        return len(self.buffers)

    def print_buffers_len(self):
        print("Length of all buffers:", len(self.buffers))
