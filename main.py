from utils.read_batch import read_recorded_data
import os

# Insert the file name here and start the program
def main(file_name="test_raw.txt"):
    file_path = "Sandbox/Samples/" + file_name
    _, file_extension = os.path.splitext(file_path)
    read_recorded_data(file_path, file_extension.lstrip('.'))

if __name__ == "__main__":
    main()
