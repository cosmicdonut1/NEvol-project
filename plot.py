import sys

import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTextEdit, QTableView
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QStandardItemModel, QStandardItem

import config

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# from plotting_logic import autoscale_y
from analyze import analyze_bandpower


class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)


def dataframe_to_model(df):
    model = QStandardItemModel(df.shape[0], df.shape[1])
    model.setHorizontalHeaderLabels(df.columns)

    for row in range(df.shape[0]):
        for column in range(df.shape[1]):
            item = QStandardItem(str(df.iat[row, column]))
            model.setItem(row, column, item)

    return model


class MainWindow(QMainWindow):

    def __init__(self, buffer_manager, current_buffer):
        super(MainWindow, self).__init__()

        # Setting up the Layout
        self.setWindowTitle('BCI Visualizations')
        self.setGeometry(100, 100, 1080, 720)

        # Creating 2 sections of the UI
        self.layout1 = QHBoxLayout()
        self.layout2 = QVBoxLayout()
        # self.layout3 = QVBoxLayout()

        self.layout1.setContentsMargins(0, 0, 0, 0)
        self.layout1.setSpacing(10)

        # Adding 2 widgets to left layout
        # self.layout2.addWidget(Color('light gray')

        self.layout1.addLayout(self.layout2)

        self.layout1.addWidget(Color('gray'))

        # self.layout3.addWidget(Color('red'))
        # self.layout3.addWidget(Color('purple'))

        # self.layout1.addLayout( layout3 )

        self.widget = QWidget(self)
        self.widget.setLayout(self.layout1)
        self.setCentralWidget(self.widget)

        # Configuring the plots
        self.fig, self.axs = plt.subplots(8, figsize=(12, 16), sharex=True)
        self.canvas_bp = FigureCanvas(self.fig)

        self.layout2.addWidget(self.canvas_bp)
        self.layout2.addWidget(Color('light gray'))

        # self.text_edit = QTextEdit()
        # self.layout1.addWidget(self.text_edit)

        self.table_view = QTableView()
        self.layout1.addWidget(self.table_view)

        # Loading relevant settings
        self.channels = config.device_details['channels']

        # Loading the relevant data
        self.dataframe = pd.DataFrame()

        self.buffers = buffer_manager
        self.curr_buffer_data = self.buffers.get_data(mode='last')
        self.all_buffers_data = self.buffers.get_data(mode='all')

        self.engagement_indices = []
        self.model = dataframe_to_model(self.dataframe)
        self.table_view.setModel(self.model)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(2000)  # Trigger update_ui every 2 seconds

    def plot_signals(self):
        # Simulate getting new data
        self.curr_buffer_data = self.buffers.get_data(mode='last')

        data = self.curr_buffer_data.iloc[:, :8]
        timestamps = self.curr_buffer_data['timestamp'] * 1e4

        # Clear the current axes to prevent over-plotting
        for ax in self.axs:
            ax.clear()

        # Plot the data
        for i in range(8):
            self.axs[i].plot(timestamps, data.iloc[:, i], label=self.channels[i])
            self.axs[i].set_ylabel(f'{self.channels[i]} values')
            self.axs[i].legend(loc='upper right')
            self.axs[i].margins(0.2, 0.2)
            self.axs[i].set_xlim([np.min(timestamps), np.max(timestamps)])
            # self.axs[i].set_aspect('auto')
            # autoscale_y(self.axs[i])

        # Set the x-axis label and title on the bottom subplot
        self.axs[-1].set_xlabel('Timestamp')
        self.fig.suptitle('Signals Over Time')

        # Redraw the canvas
        self.canvas_bp.draw()

    def plot_bp(self):
        # Updating the bandpower table
        self.curr_buffer_data = self.buffers.get_data(mode='last')

        signal_type = "baseline"
        ls_rel_channels = ['Fz', 'Cz', 'Pz', 'Oz']
        ls_rel_bands = ['theta', 'alpha', 'beta']

        self.dataframe = analyze_bandpower(self.curr_buffer_data, self.channels, signal_type,
                                           ls_rel_channels, ls_rel_bands)

        # Calculate Engagement Index and add to the DataFrame
        self.dataframe['Engagement Index'] = self.dataframe['beta'] / (
                self.dataframe['theta'] + self.dataframe['alpha'])

        # Store the Engagement Index over time
        # Assuming you want the mean Engagement Index across channels
        self.engagement_indices.append(
            self.dataframe['Engagement Index'].mean())

        self.model = dataframe_to_model(self.dataframe)
        self.table_view.setModel(self.model)

        self.plot_engagement_index()

    def update_ui(self):
        print("Updating all UI elements...")
        self.plot_signals()
        # self.plot_bp()

    def plot_engagement_index(self):
        # Clear the current axes to prevent over-plotting
        self.axs[-1].clear()

        # Plot the Engagement Index
        self.axs[-1].plot(self.engagement_indices, label='Engagement Index')
        self.axs[-1].set_ylabel('Engagement Index')
        self.axs[-1].legend(loc='upper right')
        self.axs[-1].margins(0.2, 0.2)
        self.axs[-1].set_xlim([0, len(self.engagement_indices)])

        # Redraw the canvas
        self.canvas_bp.draw()
