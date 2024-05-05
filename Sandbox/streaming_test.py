import argparse
import time
import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

def main():
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
            time.sleep(1)  # Delay for stability
        
            data = board.get_current_board_data(256)  # Get current data from board
            df = pd.DataFrame(np.transpose(data))

            # Select only every second row starting from the first (0, 2, 4, 6, ...)
            df_filtered = df.iloc[::2, :]

            accumulated_data = pd.concat([accumulated_data, df_filtered], ignore_index=True)

            # Print the filtered data frame in console
            print(df_filtered.to_string(index=False, header=False))


    except KeyboardInterrupt:
        print('Stream stopped by user.')

    finally:
        board.stop_stream()
        board.release_session()

        # Save all the accumulated data to CSV once the stream is stopped
        accumulated_data.to_csv('simulated_stream_data.csv', index=False)
        print('Full data has been saved to "simulated_stream_data.csv".')
        print('Session ended.')

if __name__ == "__main__":
    main()