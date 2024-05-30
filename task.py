"""Send event triggers in PsychoPy with LabStreamingLayer.

In this example, the words "forward" and "reverse" alternate on the screen, and
an event marker is sent with the appearance of each word.

TO RUN: open in PyschoPy Coder and press 'Run'. Or if you have the psychopy
Python package in your environment, run `python task.py` in command line.
------------
"""
from psychopy import core, visual, event
from pylsl import StreamInfo, StreamOutlet


def execute_task():
    # Set up LabStreamingLayer stream.
    info = StreamInfo(name='task_stream', type='Markers', channel_count=1,
                      channel_format='int32', source_id='task_stream_001')
    outlet = StreamOutlet(info)  # Broadcast the stream.

    # This is not necessary but can be useful to keep track of markers and the
    # events they correspond to.
    markers = {
        'forward_start': [1],
        'forward_end': [2],
        'reverse_start': [3],
        'reverse_end': [4],
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
    win = visual.Window([800, 600], allowGUI=True, monitor='testMonitor',  units='deg', color=[1, 1, 1])
    forward = visual.TextStim(win, text="say forward", color='green')
    reverse = visual.TextStim(win, text="say reverse", color='red')

    outlet.push_sample(markers['task_start'])
    for i in range(3):
        if not i % 2:
            outlet.push_sample(markers['forward_start'])
            forward.draw()
            # # Experiment with win.callOnFlip method. See Psychopy window docs.
            win.callOnFlip(outlet.push_sample, markers['forward_end'])
            # win.flip()
            # outlet.push_sample(markers['forward_end'])
        else:
            outlet.push_sample(markers['reverse_start'])
            reverse.draw()
            # # Experiment with win.callOnFlip method. See Psychopy window docs.
            win.callOnFlip(outlet.push_sample, markers['reverse_end'])
            # win.flip()
            # outlet.push_sample(markers['reverse_end'])
        if 'escape' in event.getKeys():  # Exit if user presses escape.
            break
        core.wait(2.5)  # Display text for 1.0 second.
        win.flip()
        outlet.push_sample(markers['rest_start'])
        core.wait(1)  # ISI of 0.5 seconds.
        outlet.push_sample(markers['rest_end'])
    outlet.push_sample(markers['task_end'])
    win.close()
    core.quit()


if __name__ == '__main__':
    execute_task()