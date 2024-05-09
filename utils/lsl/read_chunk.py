"""how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
import matplotlib as plt

def read_chunk():
    
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream("name", "UN-2023.04.61")

    if len(streams) == 0:
        print("No stream found.")
        return
    
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    # plt = create_empty_plot()

    duration = 2
    sampling_rate = 125
    buffer = []
    buffer_size = int(duration*sampling_rate)
    
    try:
        while True:
            # To get a new sample (you can also omit the timestamp part if you're not interested in it)
            # sample, timestamp = inlet.pull_sample()

            # To get a new chunk
            sample, timestamp = inlet.pull_chunk()

            if timestamp:
                print("Acquired sample\n", sample)


    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully...")
        sys.exit(0)