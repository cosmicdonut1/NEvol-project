import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

def simulate_data():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Start streaming, press Ctrl+C to stop...')

    # Create empty DataFrame for collecting data
    # eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    accumulated_data = pd.DataFrame()

    try:
        while True:
            #time.sleep(1)  # Delay for stability
            data = board.get_current_board_data(256)  # Get current data from board
            df = pd.DataFrame(np.transpose(data))



            df_filtered = df.iloc[::2, :].copy()
            df_filtered['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            accumulated_data = pd.concat([accumulated_data, df_filtered], ignore_index=True)

            # Print the filtered data frame in console
            print(df_filtered.to_string(index=False, header=False))


    except KeyboardInterrupt:
        print('Stream stopped by user.')

    finally:
        board.stop_stream()
        board.release_session()

        # Define header.
        header = (
            #"%OpenBCI Raw EXG Data\n"
            #"%Number of channels = 16\n"
            #"%Sample Rate = 125 Hz\n"
            #"%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
            "Sample Index, EXG Channel 0, EXG Channel 1, EXG Channel 2, EXG Channel 3, EXG Channel 4, EXG Channel 5, "
            "EXG Channel 6, EXG Channel 7, EXG Channel 8, EXG Channel 9, EXG Channel 10, EXG Channel 11, EXG Channel 12, "
            "EXG Channel 13, EXG Channel 14, EXG Channel 15, Accel Channel 0, Accel Channel 1, Accel Channel 2, Other, "
            "Other, Other, Other, Other, Other, Other, Analog Channel 0, Analog Channel 1, Analog Channel 2, Timestamp, Other, "
            "Timestamp (Formatted)\n"
        )

        # File name
        file_name = 'simulated_stream_data.csv'

        # Convert DataFrame to string and append header
        accumulated_data_str = accumulated_data.to_csv(index=False, header=False)
        data_str = header + accumulated_data_str

        # Write to file
        #with open(file_name, 'w') as f:
            #f.write(data_str)

        with open(file_name, 'w', newline='') as f:
            f.write(data_str)

        print('Full data has been saved to "{}".'.format(file_name))
        print('Session ended.')