
import argparse
import time
import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

import os
import time

def create_brainflow_board():
        # Initialize BrainFlow board
        params = BrainFlowInputParams()
        board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        return board

def list_to_dataframe(data_list):
        # Split each element of the list by comma and create a list of lists
        split_data = [line.split(',') for line in data_list]
        
        # Use the first element of the split_data list as the header
        header = split_data[0]

        # Remove the header from the split_data list
        data = split_data[1:]

        # Convert the list of lists to a DataFrame with the specified header
        df = pd.DataFrame(data, columns=header)

        return df

# Read EEG data from CSV file using Pandas or other libraries
def read_data_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    return data

# Read EEG data from TXT file as shown in Samples/test_raw.txt
def read_data_from_txt(txt_file):
    header = None
    data_lines = []

    with open(txt_file, 'r') as file:
        lines = file.readlines()

        # Find header and data lines
        for line in lines:
            if line.startswith('%OpenBCI Raw EXG Data'):
                header = line.strip()
            elif not line.startswith('%'):
                data_lines.append(line.strip())

    return header, data_lines

def print_rows_in_intervals(df, sampling_rate):
    num_of_seconds = len(df) // sampling_rate

    for second in range(num_of_seconds):
        start_index = second * sampling_rate
        end_index = start_index + sampling_rate

        print("Rows from", start_index, "to", end_index - 1)
        print(df.iloc[start_index:end_index])

        time.sleep(1)  # Wait for 1 second before printing the next interval

def read_recorded_data(file_path, file_type):

    board = create_brainflow_board()
    board.get_board_descr(BoardIds.CYTON_DAISY_BOARD)

    if file_type == "txt":
        txt_header, data_lines = read_data_from_txt(file_path)
        df = list_to_dataframe(data_lines)

        # drop the last column "Timstamp(Formatted)"
        df.drop(df.columns[-1], axis=1, inplace=True)

        # remove first two rows from df - as I see sample index is mentioned as 0 and 46.0 which is strange
        # df = df.iloc[2:]

    elif file_type == "csv":
        data = read_data_from_csv(file_path)
        header = data.columns
        data_lines = data.values.tolist()
        df = pd.DataFrame(data_lines)
        
        # # Customized for Cyton Daisy, should be changed if device used is different
        # df.columns = ['Sample Index', ' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2',
        # ' EXG Channel 3', ' EXG Channel 4', ' EXG Channel 5', ' EXG Channel 6',
        # ' EXG Channel 7', ' EXG Channel 8', ' EXG Channel 9', ' EXG Channel 10',
        # ' EXG Channel 11', ' EXG Channel 12', ' EXG Channel 13',
        # ' EXG Channel 14', ' EXG Channel 15', ' Accel Channel 0',
        # ' Accel Channel 1', ' Accel Channel 2', ' Other', ' Other', ' Other',
        # ' Other', ' Other', ' Other', ' Other', ' Analog Channel 0',
        # ' Analog Channel 1', ' Analog Channel 2', ' Timestamp', ' Other']

    df.head()

    board = create_brainflow_board()
    sampling_rate = board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
    num_of_seconds = len(df) / sampling_rate

    print("Total number of rows in df: ", len(df))
    print("Sampling Rate from the board: ", sampling_rate)
    print("Total seconds of data from the df: ", num_of_seconds)

    print_rows_in_intervals(df, sampling_rate)