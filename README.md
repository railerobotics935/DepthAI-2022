# DepthAI-2022
OAK-D DepthAI camera support. Contains Python code for RPi host and utilities for training the OAK-D camera

Folder CVAT contains the label settings for annotating images. Load this under "raw" in CVAT when creating an annotation task.
see cvat.org

Folder RPi_script contains the scripts that have been developed and tested to run as uploaded applications in the FRC WPILIBPI image for a Raspberry PI coprocessor. The WPILIBPI needs to get the DepthAI libraries installed. The latest in early March 2022 was DepthAI library version 2.15.0. The purpose of the RPi coprocesser is to start and run the OAK-D camera with a custom trained AI model and convert the video and the detected objects tracking streams and data via NetworkTables to the RoboRIO.

Checklist for the RPi coprocessor:

- The RPi gets power from a VRM, use one of the 5V 2A ports. **Be careful to connect the power cable to the correct pins on the RPi extension connector**.
- The RPi network cable goes to the network port on the radio opposite of the power plug on the radio.
- After the RPi is powered up and a laptop is connected to the robot radio, then the RPi serves a webpage that is accessible at URL: **http://wpilibpi.local**
- If the URL doesn't work try http://10.9.35.63 (63 being the address dynamically assigned to the RPi, this may be another address in the 10.9.35.x range)
- The OAK-D Lite camera is connected via a USB-C to USB3 cable to a USB3 port on the RPi.

When the webpage served by the RPi is visible, a new vision application can be uploaded: last tab, upload a .py file like the "frc-test-07.py". The uploaded vision application is automatically started when the RPi boots and it restarts if it crashes or is overwritten by a new vision application. The application can also be manually started and stopped: see second tab with "start", "stop" buttons and a console output log switch. Enabling the console output is useful for debugging vision applications.

The vision application writted in "frc-test-07.py" can act as NetworkTables server or client. When using the vision application on the robot it has to be in "Client" mode with team number 935 specified to find the NetworkTables server running in the RoboRIO. For benchtest purposes the application can be set to "Server" mode, this is useful when connecting directly from a PC to the RPi and using e.g. Shuffleboard on the PC pointed to the RPi as NetworkTables server.

# Tips and Tricks for Annotation and Machine Learning
- Export format for CAVT is PASCAL VOC 1.1
