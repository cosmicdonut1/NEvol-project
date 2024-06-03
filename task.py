"""Send event triggers in PsychoPy with LabStreamingLayer.

In this example, the words "left" and "right" alternate on the screen, and
an event marker is sent with the appearance of each word.

TO RUN: open in PyschoPy Coder and press 'Run'. Or if you have the psychopy
Python package in your environment, run `python task.py` in command line.
------------
"""
from psychopy import core, visual, event
from pylsl import StreamInfo, StreamOutlet


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

        outlet.push_sample(markers['task_start'])
        wait.draw()
        core.wait(1.5)
        win.flip()
        for i in range(iterations):
            if not i % 2:
                outlet.push_sample(markers['left_start'])
                left.draw()
                # # Experiment with win.callOnFlip method. See Psychopy window docs.
                win.callOnFlip(outlet.push_sample, markers['left_end'])
                # win.flip()
                # outlet.push_sample(markers['left_end'])
            else:
                outlet.push_sample(markers['right_start'])
                right.draw()
                # # Experiment with win.callOnFlip method. See Psychopy window docs.
                win.callOnFlip(outlet.push_sample, markers['right_end'])
                # win.flip()
                # outlet.push_sample(markers['right_end'])
            if 'escape' in event.getKeys():  # Exit if user presses escape.
                break
            core.wait(1.5)  # Display text for 1.0 second.
            win.flip()
            outlet.push_sample(markers['rest_start'])
            core.wait(1)  # ISI of 0.5 seconds.
            outlet.push_sample(markers['rest_end'])
        outlet.push_sample(markers['task_end'])
        win.close()
        core.quit()


if __name__ == '__main__':
    execute_train_task()
