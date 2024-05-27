from threading import Thread, Event

from buffer import Buffer
from lsl_manager import *

import time
import numpy as np
# import pandas as pd
from plot import plotEEGData
from analyze import analyze_signal, classify_eyeblinks


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

    if bool_s_stream_status:
        signal_thread = Thread(target=read_signal_stream, args=("UN-2023.04.61", buffer, stop_event))
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
