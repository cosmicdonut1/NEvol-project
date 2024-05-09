import math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from collections import deque
from pylsl import StreamInlet, resolve_stream

# Basic parameters for the plotting window
plot_duration = 5  # how many seconds of data to show
update_interval = 60  # ms between screen updates
pull_interval = 500  # ms between each pull operation

class DataInlet:
    """Class to represent an inlet with continuous, multi-channel data"""

    def __init__(self, inlet, plt):
        self.inlet = inlet
        self.channel_count = inlet.info().channel_count()
        bufsize = 2 * math.ceil(inlet.info().nominal_srate() * plot_duration)
        self.buffer = np.empty((bufsize, self.channel_count), dtype=np.float32)
        self.data_buffer = deque(maxlen=bufsize)
        empty = np.array([])
        self.curves = [pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True) for _ in range(self.channel_count)]
        for curve in self.curves:
            plt.addItem(curve)

    def pull_and_plot(self, plot_time, plt):
        _, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=self.buffer.shape[0], dest_obj=self.buffer)
        if ts:
            ts = np.asarray(ts)
            y = self.buffer[0:ts.size, :]
            this_x = None
            old_offset = 0
            new_offset = 0
            for ch_ix in range(self.channel_count):
                if ch_ix == 0:
                    old_offset = self.data_buffer.__len__()
                    new_offset = ts.searchsorted(plot_time)
                    this_x = np.hstack((np.linspace(plot_time - plot_duration, plot_time, num=old_offset), ts[new_offset:]))
                this_y = np.hstack((self.data_buffer, y[new_offset:, ch_ix] - ch_ix))
                self.curves[ch_ix].setData(this_x, this_y)
            self.data_buffer.extend(ts)


def main():
    # Resolve LSL stream
    print("Looking for an EEG stream...")
    streams = resolve_stream("name", "UN-2023.04.61")

    if len(streams) == 0:
        print("No stream found.")
        return
    

    inlet = StreamInlet(streams[0])



    # Initialize DataInlet
    data_inlet = DataInlet(inlet, plt)

    def scroll():
        # Move the view so the data appears to scroll
        fudge_factor = pull_interval * 0.002
        plot_time = QtGui.QDateTime.currentDateTime().toMSecsSinceEpoch() / 1000.0
        plt.setXRange(plot_time - plot_duration + fudge_factor, plot_time - fudge_factor)

    def update():
        # Pull and plot data
        mintime = QtGui.QDateTime.currentDateTime().toMSecsSinceEpoch() / 1000.0 - plot_duration
        data_inlet.pull_and_plot(mintime, plt)

    # Create a timer to move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # Create a timer to pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    # Start Qt event loop
    QtGui.QGuiApplication.instance().exec_()


if __name__ == "__main__":
    main()
