import UnicornPy
import struct
import pandas as pd
import numpy as np

"""
Acquires data from a Unicorn device for a specified duration.

This function initializes the data acquisition process by getting available devices,
selecting a device, and configuring the acquisition parameters. It then starts the
acquisition, receives and processes data from the device, and stops the acquisition.
The acquired data is stored in a list of numpy arrays and saved to a CSV file.

Parameters:
None

Returns:
None

Raises:
- UnicornPy.DeviceException: If there is an error with the Unicorn device.
- Exception: If an unknown error occurs.

"""
def acquire_data_from_unicorn_device():
    # Specifications for the data acquisition.
    #-------------------------------------------------------------------------------------
    TestsignaleEnabled = False;
    FrameLength = 1;
    AcquisitionDurationInSeconds = 5;

    unicorn_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    unicorn_fs = UnicornPy.SamplingRate
    # DataFile = "data.csv";

    np_all_samples_values = []

    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    try:
        # Get available devices.
        #-------------------------------------------------------------------------------------

        # Get available device serials.
        deviceList = UnicornPy.GetAvailableDevices(True)

        if len(deviceList) <= 0 or deviceList is None:
            raise Exception("No device available.Please pair with a Unicorn first.")

        # Print available device serials.
        print("Available devices:")
        i = 0
        for device in deviceList:
            print("#%i %s" % (i,device))
            i+=1

        # Request device selection.
        print()
        deviceID = int(input("Select device by ID #"))
        if deviceID < 0 or deviceID > len(deviceList):
            raise IndexError('The selected device ID is not valid.')

        # Open selected device.
        #-------------------------------------------------------------------------------------
        print()
        print("Trying to connect to '%s'." %deviceList[deviceID])
        device = UnicornPy.Unicorn(deviceList[deviceID])
        print("Connected to '%s'." %deviceList[deviceID])
        print()

        # Create a file to store data.
        # file = open(DataFile, "wb")

        # Initialize acquisition members.
        #-------------------------------------------------------------------------------------
        numberOfAcquiredChannels= device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()

        # Print acquisition configuration
        print("Acquisition Configuration:");
        print("Sampling Rate: %i Hz" %UnicornPy.SamplingRate);
        print("Frame Length: %i" %FrameLength);
        print("Number Of Acquired Channels: %i" %numberOfAcquiredChannels);
        print("Data Acquisition Length: %i s" %AcquisitionDurationInSeconds);
        print();

        # Allocate memory for the acquisition buffer.
        receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        try:
            # Start data acquisition.
            #-------------------------------------------------------------------------------------
            device.StartAcquisition(TestsignaleEnabled)
            print("Data acquisition started.")

            # Calculate number of get data calls.
            numberOfGetDataCalls = int(AcquisitionDurationInSeconds * UnicornPy.SamplingRate / FrameLength);
        
            # Limit console update rate to max. 25Hz or slower to prevent acquisition timing issues.                   
            consoleUpdateRate = int((UnicornPy.SamplingRate / FrameLength) / 25.0);
            if consoleUpdateRate == 0:
                consoleUpdateRate = 1

            # Acquisition loop.
            #-------------------------------------------------------------------------------------
            for i in range (0,numberOfGetDataCalls):
                # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
                device.GetData(FrameLength,receiveBuffer,receiveBufferBufferLength)

                # Unpack the received bytes assuming they represent 4-byte floats (change format according to your data)
                values = struct.unpack('f' * (len(receiveBuffer) // 4), receiveBuffer)

                # Print the received values
                print("Received values:", values[:8])
                np_all_samples_values.append(values[:8])
                # Write data to file.
                # file.write(values)

                # Update console to indicate that the data acquisition is running.
                if i % consoleUpdateRate == 0:
                    print('.',end='',flush=True)

            # Stop data acquisition.
            #-------------------------------------------------------------------------------------
            device.StopAcquisition();
            print()
            print("Data acquisition stopped.");

        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print("An unknown error occured. %s" %e)
        finally:
            # release receive allocated memory of receive buffer
            del receiveBuffer

            #close file
            # file.close()

            # Close device.
            #-------------------------------------------------------------------------------------
            del device
            print("Disconnected from Unicorn")

            data_array = np.array(np_all_samples_values)
            print(data_array.shape)

            df_data_array = pd.DataFrame(data_array, columns=unicorn_channels)
            print("Now saving the dataframe to a csv file")

            df_data_array.to_csv('data.csv', index=False)

    except UnicornPy.DeviceException as e:
        print(e)
    except Exception as e:
        print("An unknown error occured. %s" %e)


    
    input("\n\nPress ENTER key to exit")