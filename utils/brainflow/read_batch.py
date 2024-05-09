import time
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def initialize_board(brainflow_input_params):
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, brainflow_input_params)
    return board


def read_data_from_txt(txt_file_path):
    with open(txt_file_path, 'r') as file:
        # Remove header from data lines
        data_lines = [line.strip() for line in file.readlines() if not line.startswith('%')]
    return data_lines


def read_data_from_csv(csv_file_path):
    return pd.read_csv(csv_file_path).values.tolist()


def fetch_data(file_path, file_type):
    if file_type == 'txt':
        data_lines = read_data_from_txt(file_path)

        if len(data_lines) > 0:
            # Remove "Timestamp" column
            data = [line.split(',') for line in data_lines]
            df = pd.DataFrame(data[1:], columns=data[0])
    elif file_type == 'csv':
        df = pd.DataFrame(read_data_from_csv(file_path))
    return df


def print_rows_in_intervals(df, sampling_rate):
    num_intervals = len(df) // sampling_rate

    for interval in range(num_intervals):
        start_index = interval * sampling_rate
        end_index = start_index + sampling_rate
        #print("Rows from", start_index, "to", end_index - 1)
        print(df.iloc[start_index:end_index].to_string(index=False))
        time.sleep(1)  # Wait for 1 second before printing the next interval

def read_recorded_data(file_path, file_type):
    brainflow_input_params = BrainFlowInputParams()
    board = initialize_board(brainflow_input_params)

    board.get_board_descr(BoardIds.CYTON_DAISY_BOARD)
    df = fetch_data(file_path, file_type)

    #print("Total number of rows in df: ", len(df))

    sampling_rate = board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
    #print("Sampling Rate from the board: ", sampling_rate)

    num_seconds = len(df) / sampling_rate
    #print("Total seconds of data from the df: ", num_seconds)
    print_rows_in_intervals(df, sampling_rate)