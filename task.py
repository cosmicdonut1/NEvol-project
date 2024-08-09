"""Send event triggers in PsychoPy with LabStreamingLayer.

In this example, the words "left" and "right" alternate on the screen, and
an event marker is sent with the appearance of each word.

TO RUN: open in PyschoPy Coder and press 'Run'. Or if you have the psychopy
Python package in your environment, run `python task.py` in command line.
------------
"""
from psychopy import core, visual, event
from pylsl import StreamInfo, StreamOutlet
import time


def execute_train_task(mode="motor_imagery", iterations=6):
    if mode == "motor_imagery":
        # Set up LabStreamingLayer stream.
        info = StreamInfo(name='task_stream', type='Markers', channel_count=1,
                          channel_format='int32', source_id='task_stream_001')
        outlet = StreamOutlet(info)  # Broadcast the stream.

        # This is not necessary but can be useful to keep track of markers and the
        # events they correspond to.
        markers = {
            'left_start': [1],
            'left_end': [2],
            'right_start': [3],
            'right_end': [4],
            'rest_start': [99],
            'rest_end': [100],
            'task_start': [-1],
            'task_end': [-2]
        }

        # Send triggers to test communication.
        # for _ in range(5):
        #     outlet.push_sample(markers['rest'])
        #     core.wait(0.5)

        # Instantiate the PsychoPy window and stimuli.
        win = visual.Window([1000, 800], allowGUI=True, monitor='testMonitor', units='deg', color=[1, 1, 1])
        left = visual.TextStim(win, text="imagine left", color='green')
        right = visual.TextStim(win, text="imagine right", color='red')
        wait = visual.TextStim(win, text="rest", color='black')

        # Start the task and send the start marker.
        outlet.push_sample(markers['task_start'])
        time_start = time.time()
        print(f"Task started at: {time_start}")

        for i in range(iterations // 2):
            # Display "left" and record markers.
            outlet.push_sample(markers['left_start'])
            left.draw()
            win.flip()
            time_left_start = time.time()
            print(f"Left start at: {time_left_start}")

            core.wait(3)  # Show for 3 seconds.

            outlet.push_sample(markers['left_end'])
            win.flip()
            time_left_end = time.time()
            print(f"Left end at: {time_left_end}")

            # Display "rest" and record markers.
            outlet.push_sample(markers['rest_start'])
            wait.draw()
            win.flip()
            time_rest_start = time.time()
            print(f"Rest start at: {time_rest_start}")

            core.wait(1)  # Show for 1 second.

            outlet.push_sample(markers['rest_end'])
            win.flip()
            time_rest_end = time.time()
            print(f"Rest end at: {time_rest_end}")

            # Display "right" and record markers.
            outlet.push_sample(markers['right_start'])
            right.draw()
            win.flip()
            time_right_start = time.time()
            print(f"Right start at: {time_right_start}")

            core.wait(3)  # Show for 3 seconds.

            outlet.push_sample(markers['right_end'])
            win.flip()
            time_right_end = time.time()
            print(f"Right end at: {time_right_end}")

            # Display "rest" again and record markers.
            outlet.push_sample(markers['rest_start'])
            wait.draw()
            win.flip()
            time_rest_start_2 = time.time()
            print(f"Rest start (2) at: {time_rest_start_2}")

            core.wait(1)  # Show for 1 second.

            outlet.push_sample(markers['rest_end'])
            win.flip()
            time_rest_end_2 = time.time()
            print(f"Rest end (2) at: {time_rest_end_2}")

        # End the task and send the end marker.
        outlet.push_sample(markers['task_end'])
        time_end = time.time()
        print(f"Task ended at: {time_end}")

        win.close()
        core.quit()


if __name__ == '__main__':
    execute_train_task()
