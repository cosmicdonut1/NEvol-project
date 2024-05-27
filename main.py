from threading import Thread, Event

from buffer import Buffer
from lsl_manager import *

import time
import numpy as np
# import pandas as pd
from plot import plotEEGData
from analyze import analyze_signal, classify_eyeblinks
import config

if __name__ == "__main__":
    # Initialize the variables
    info = {'start_time': time.time()}

    mode = "train"
    # mode = "predict"

    epoch_duration = config.epoch_information['duration']
    sampling_rate = config.device_details['sfreq']
    channel_names = config.device_details['channels']
    all_channels = config.device_details['total_channel_from_device']
    device_id = config.device_details['id']

    ls_markers = []
    eeg_signals = np.zeros((len(channel_names), epoch_duration * sampling_rate))

    buffer = Buffer(duration=epoch_duration, sampling_rate=sampling_rate, num_channels=all_channels)
    stop_event = Event()

    bool_s_stream_status = check_stream(device_id)

    if bool_s_stream_status:
        signal_thread = Thread(target=read_signal_stream, args=(device_id, buffer, stop_event))
        signal_thread.start()

        while True:
            # print("This is the buffer")
            df_buffer = buffer.get_plottable_data(channel_names)
            print(df_buffer)
            if df_buffer[channel_names].to_numpy().any():
                # analyze_signal('bandpower', df_buffer, channel_names)
                classify_eyeblinks('bandpower', df_buffer, channel_names)
            time.sleep(epoch_duration)

        # while True:
        #     window = RealTimePlot(buffer, channel_names, sampling_rate, epoch_duration)
        #     window.show()
        #     time.sleep(2)
        #     RealTimePlot.update_plots()
        #     sys.exit(app.exec_())
