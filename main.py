from threading import Thread, Event

from buffer import Buffer
from lsl_manager import *

import time
import numpy as np
# import pandas as pd
from plot import plotEEGData
from analyze import analyze_bandpower, classify_eyeblinks, process_imagery_data
# import task

from utils import clear_data_dumps
import config

if __name__ == "__main__":
    # Initialize the variables
    info = {'start_time': time.time()}

    # mode = "train"
    # mode = "predict"

    print("Reading configurations from config.py...")
    epoch_duration = config.epoch_information['duration']
    sampling_rate = config.device_details['sfreq']
    channel_names = config.device_details['channels']
    all_channels = config.device_details['total_channels_from_device']
    device_id = config.device_details['id']

    # ls_markers = []
    # eeg_signals = np.zeros((len(channel_names), epoch_duration * sampling_rate))

    # Create an empty buffer for storage
    buffer = Buffer(duration=epoch_duration, sampling_rate=sampling_rate, num_channels=all_channels)
    buffer.print_buffer_shape()

    stop_event = Event()

    bool_s_stream_status = check_stream(device_id)
    if bool_s_stream_status:
        print("Device ", device_id, " is connected...")

    # Execute conditions based on the task details from config file

    if config.task_details['task'] == "eyeblink" and bool_s_stream_status:
        signal_thread = Thread(target=read_signal_stream, args=(device_id, buffer, stop_event))
        signal_thread.start()

        while True:
            df_buffer = buffer.get_plottable_data(channel_names)
            print(df_buffer)
            if df_buffer[channel_names].to_numpy().any():
                classify_eyeblinks(df_buffer, channel_names)
            time.sleep(epoch_duration)

    if config.task_details['task'] == "bandpower" and bool_s_stream_status:
        print("Bandpower Analysis Initiated.")
        signal_thread = Thread(target=read_signal_stream, args=(device_id, buffer, stop_event))
        signal_thread.start()

        while True:
            df_buffer = buffer.get_plottable_data(channel_names)
            print(df_buffer)
            if df_buffer[channel_names].to_numpy().any():
                analyze_bandpower(df_buffer, channel_names, )
            time.sleep(epoch_duration)

    if config.task_details['task'] == "motor_imagery":
        print("Executing Motor Imagery | Mode = ", config.task_details['mode'])
        if config.task_details['mode'] == "train":

            bool_m_stream_status = check_stream("task_stream")
            if bool_s_stream_status and bool_m_stream_status:
                ls_markers = []
                eeg_signals = np.zeros((len(channel_names), epoch_duration * sampling_rate))

                clear_data_dumps("processed_data")

                info = {'start_time': time.time()}

                signal_thread = Thread(target=read_signal_stream, args=(device_id, buffer, stop_event))
                signal_thread.start()

                read_task_thread = Thread(target=read_task_stream, args=("task_stream", ls_markers))
                read_task_thread.start()
                read_task_thread.join()
                stop_event.set()

                info['end_time'] = time.time()

                print("Task Ended. Now preprocessing...")
                # process_imagery_data(channel_names)

        elif config.task_details['mode'] == "predict":
            pass

    else:
        print("Task or Mode invalid! Please check documentation")