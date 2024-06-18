from pylsl import StreamInlet, resolve_stream
import sys
import time
import json
import os
import numpy as np
# from utils import lsl_to_datetime
from datetime import datetime

ls_active_streams = []
# markers = []


def check_stream(device_name):
    while True:
        # first resolve an EEG stream on the lab network
        print("Waiting for stream...", device_name)

        try:
            streams = resolve_stream("name", device_name)
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting gracefully...")
            sys.exit(0)

        if len(streams) == 0:
            print("No device stream found. Retrying...")
            return False

        elif len(streams) >= 1:
            ls_active_streams.append(streams)
            print("Device stream established: ", streams)
            return True

        time.sleep(3)


def read_task_stream(task_stream_name, ls_task_markers, save_path):
    task_stream = resolve_stream("name", task_stream_name)
    inlet_task = StreamInlet(task_stream[0])
    try:
        while True:
            event_id, timestamp_markers = inlet_task.pull_sample()
            # print("task stream receiving data...")
            print("Marker Received: ", event_id, " | Marker Timestamp:", timestamp_markers)

            if timestamp_markers:
                ls_task_markers.append([event_id, timestamp_markers])
                # ls_task_markers.append([event_id, time.time()])

            # task_end marker is -2 in task.py
            if event_id == [-2]:
                # Convert list to numpy arrays and save
                event_ids, timestamps = zip(*ls_task_markers)
                event_ids = np.array(event_ids)
                timestamps = np.array(timestamps)
                filename = os.path.join(save_path, f"task_markers.npz")
                # Save to a .npz file
                np.savez(filename, event_ids=event_ids, timestamps=timestamps)
                print("Event markers saved to ", filename)
                break

    except Exception as e:
        print(f"Task stream encountered an error: {e}")
    finally:
        print("Training Task Completed!")
        inlet_task.close_stream()


def read_signal_stream(device_name, buffer, stop_event, save_path="processed_data/signal_data"):
    signal_stream = resolve_stream("name", device_name)

    # create a new inlet to read from the stream
    inlet_signal = StreamInlet(signal_stream[0])

    try:
        while not stop_event.is_set():
            sample, timestamp_signal = inlet_signal.pull_sample()

            # What if we used datetime instead
            # This will create a small-time lag | use with caution
            # The only reason this is being tried is that plots need specific datetime format
            # timestamp_signal = datetime.now()

            # To get a new chunk
            # sample, timestamp = inlet.pull_chunk()

            if timestamp_signal:
                # print("signal stream receiving data...")
                # print("Signal Received:\n", sample, "\nSignal Timestamp:", timestamp_signal)
                buffer.add_sample(sample=sample, timestamp=timestamp_signal)

    except Exception as e:
        print(f"Signal stream encountered an error: {e}")

    finally:
        print("Closing Signal Stream")
        inlet_signal.close_stream()
