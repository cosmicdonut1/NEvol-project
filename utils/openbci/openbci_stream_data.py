import time
import numpy as np
from collections import deque
from pyOpenBCI import OpenBCICyton
import xmltodict

# Set up variables
maxlen = 50
sequence = np.zeros((100, 16))  # 100 - number of samples, 16 - number of channels
fps_counter = deque(maxlen=maxlen)
last_print = time.time()


# Define the callback function for sample data
def print_raw(sample):
    global sequence, fps_counter, last_print

    # Roll the sequence array and add new sample data
    sequence = np.roll(sequence, 1, 0)
    sequence[0, :] = sample.channels_data

    # Update fps counter
    current_time = time.time()
    fps_counter.append(current_time - last_print)
    last_print = current_time

    # Print the current frame rate and first 16 channels of sequence with carriage return
    fps = 1 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

    # Create a formatted string of data
    data_str = np.array_str(sequence[0, :])

    print(f"FPS: {fps:.2f} | Data: {data_str}", flush=True)  # Print without carriage return


# Initialize the OpenBCI board with the correct COM port
board = OpenBCICyton(port='COM3', daisy=True)

# Start the data stream and set the callback function
board.start_stream(print_raw)