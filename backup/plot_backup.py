
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
# from datetime import datetime, timedelta
import time

import config


class MainWindow(QMainWindow):
    def __init__(self, buffer_manager, current_buffer):
        super(MainWindow, self).__init__()

        self.buffers = buffer_manager
        self.curr_buffer_data = self.buffers.get_data(mode='last')
        self.all_buffers_data = self.buffers.get_data(mode='all')

        self.channels = config.device_details['channels']

        self.setWindowTitle('Brain Computer Interface Visualizations')
        self.setGeometry(100, 100, 1080, 720)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.fig, self.axs = plt.subplots(8, figsize=(7, 15), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        self.text_edit = QTextEdit()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(2000)  # Trigger update_ui every 2 seconds

    def update_ui(self):
        print("Updating UI")

        # Simulate getting new data
        self.curr_buffer_data = self.buffers.get_data(mode='last')

        data = self.curr_buffer_data.iloc[:, :8]
        timestamps = self.curr_buffer_data['timestamp']

        # Clear the current axes to prevent over-plotting
        for ax in self.axs:
            ax.clear()

        # Plot the data
        for i in range(8):
            self.axs[i].plot(timestamps, data.iloc[:, i], label=self.channels[i])
            self.axs[i].set_ylabel(f'{self.channels[i]} values')
            self.axs[i].legend(loc='upper right')

        # Set the x-axis label and title on the bottom subplot
        self.axs[-1].set_xlabel('Timestamp')
        self.fig.suptitle('Signals Over Time')

        # Redraw the canvas
        self.canvas.draw()

        self.text_edit.setPlainText(data.to_string(index=False))



