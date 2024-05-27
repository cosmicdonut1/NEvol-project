import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import sys
import numpy as np
from utils import lsl_to_datetime

def plotEEGData(df_buffer, channel_names):
    app = QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title="Real-Time EEG Signal Visualization")
    win.resize(1000, 600)
    win.setWindowTitle('EEG Plot')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    plot_widgets = {}
    curves = {}

    # Create plots for each channel
    for channel in channel_names:
        plot_widgets[channel] = win.addPlot(title=channel)
        curves[channel] = plot_widgets[channel].plot(pen='y')
        win.nextRow()

    for channel in channel_names:
        y_data = df_buffer[channel]
        x_data = df_buffer['timestamp'].apply(lsl_to_datetime)
        curves[channel].setData(x=x_data, y=y_data)

    sys.exit(app.exec_())
