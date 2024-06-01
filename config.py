# Venkat's Unicorn
device_details = dict(
    id="UN-2023.04.61",
    sfreq=125,
    total_channels_from_device = 17,
    relevant_channels_from_device = 8,
    channels=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
)
epoch_information = dict(
    # buffer size in seconds
    duration=1,
)

task_details = dict(
    task="motor_imagery",
    # task="eyeblink",
    # task="bandpower",
    bandpower_reference_channels=['Fz', 'Cz', 'Pz'],
    eyeblink_eog_channel = 'Fz',
    mode="train"
)

# Kseniia's OpenBCI
# device_details = dict(
#     id="UN-2023.04.61",
#     sfreq=125,
#     total_channels_from_device = 17,
#     relevant_channels_from_device = 17,
#     channels=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
# )
# epoch_information = dict(
#     # buffer size in seconds
#     duration=2,
# )
