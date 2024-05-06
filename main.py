import argparse
import time
import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

from utils.simulate_synthetic import simulate_data
from utils.read_batch import read_recorded_data
# write a util function to read real-time data from the device

def main():

    # Select one of the two modes for data input
    # 1. Read recorded data from CSV file
    # 2. Simulate synthetic data real-time

    # 1 is recorded data
    # 2 is synthetic data
    mode = 1

    if mode == 1:
        # file_type = 'txt'
        # file_path = 'Sandbox/Samples/test_raw.txt'
        
        file_type = 'csv'
        file_path = "Sandbox/Samples/test_raw_2.csv"

        read_recorded_data(file_path, file_type)

    if mode == 2:
        simulate_data()

if __name__ == "__main__":
    main()
