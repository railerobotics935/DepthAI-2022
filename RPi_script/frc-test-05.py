#!/usr/bin/env python3
#
# based on: https://docs.luxonis.com/projects/api/en/v2.1.0.0/samples/26_1_spatial_mobilenet/
# and: https://docs.luxonis.com/projects/api/en/latest/samples/ObjectTracker/spatial_object_tracker/#spatial-object-tracker-on-rgb
# updated to work on FRC 2022 WPILibPi image and to be uploaded as a vision application
# communicating with shuffleboard and RoboRIO through NetworkTables and CameraServer
# Jaap van Bergeijk, 2022

#from operator import truediv
from pathlib import Path
from cscore import CameraServer
from networktables import NetworkTables

import sys
# import blobconverter # use a pre-converted blob instead so this can run on a RO file-system
import cv2
import depthai as dai
import numpy as np
import time

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

syncNN = True
streamDepth = False

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

# Create a CameraServer for ShuffleBoard visualization
cs = CameraServer.getInstance()
cs.enableLogging()

# Width and Height have to match Neural Network settings
width = 300
height = 300
# output_stream_front_cam = cs.putVideo("FrontCam", width, height) 
output_stream_front_nn = cs.putVideo("FrontNN", width, height)
if streamDepth:
    output_stream_front_depth = cs.putVideo("FrontDepth", width, height)

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = dai.Pipeline()

# First, we want the Color camera as the output
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
objectTracker = pipeline.createObjectTracker()

xoutRgb = pipeline.createXLinkOut()
#xoutNN = pipeline.createXLinkOut()
#xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
if streamDepth:
    xoutDepth = pipeline.createXLinkOut()
xoutTracker = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
#xoutNN.setStreamName("detections")
#xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
if streamDepth:
    xoutDepth.setStreamName("depth")
xoutTracker.setStreamName("tracklets")

#colorCam.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
colorCam.setPreviewSize(width, height)  # 300x300 will be the preview frame size, available as 'preview' output of the node
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
#spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
#spatialDetectionNetwork.setDepthUpperThreshold(5000)
spatialDetectionNetwork.setDepthUpperThreshold(15000)

objectTracker.setDetectionLabelsToTrack([9, 15, 20])  # track chairs, persons and tvmonitors
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
objectTracker.out.link(xoutTracker.input)

if (syncNN):
#    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)

#spatialDetectionNetwork.out.link(xoutNN.input)
#spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
#spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# NetworkTables commmunication, setup as server for PC based testing without a RoboRIO
NetworkTables.initialize()
sd=NetworkTables.getTable("SmartDashboard")

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with dai.Device(pipeline, True) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
#    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
#    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    if streamDepth:
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    trackletsQueue = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    detections = []

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    while True:
        inPreview = previewQueue.get()
#        inNN = detectionNNQueue.get()
        if streamDepth:
            depth = depthQueue.get()
        track = trackletsQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()
        if streamDepth:
            depthFrame = depth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        trackletsData = track.tracklets
        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)

            ssd=sd.getSubTable(f"tracked_object_{t.id}")

            ssd.putString("label", label)
            ssd.putString("status", t.status.name)
#            sd.putNumber("confidence", int(detection.confidence * 100))
            ssd.putNumberArray("x,y,z", [int(t.spatialCoordinates.x), int(t.spatialCoordinates.y), int(t.spatialCoordinates.z)])

#        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color)

        # After all the drawing is finished, we show the frame on the screen
        # we can't do that on a headless RPi....           cv2.imshow("preview", frame)
        # Instead publish to CameraServer output stream for NetworkTables or MJPEG http stream
        output_stream_front_nn.putFrame(frame)
        if streamDepth:
            output_stream_front_depth.putFrame(depthFrameColor)
#        cv2.imshow("depth", depthFrameColor)
#        cv2.imshow("rgb", frame)

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break