# DepthAI-2022
OAK-D DepthAI camera support. Contains Python code for RPi host and utilities for training the OAK-D camera

Folder CVAT contains the label settings for annotating images. Load this under "raw" in CVAT when creating an annotation task.
see cvat.org

Folder RPi_script contains the scripts that have been developed and tested to run as uploaded applications in the FRC WPILIBPI image for a Raspberry PI coprocessor. The WPILIBPI needs to get the DepthAI libraries installed. The latest in early March 2022 was DepthAI library version 2.15.0. The purpose of the RPi coprocesser is to start and run the OAK-D camera with a custom trained AI model and convert the video and the detected objects tracking streams and data via NetworkTables to the RoboRIO.