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
from networktables import NetworkTablesInstance

import json
import time
import sys
import cv2
import depthai as dai
import numpy as np

configFile = "/boot/frc.json"

team = None
server = False

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    return True

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)
        ntinst.startDSClient()
    sd=ntinst.getTable("SmartDashboard")

    # MobilenetSSD label texts
    labelMap = ["background", "blue ball", "person", "red ball", "robot"]

    syncNN = True
    streamDepth = False

    # Static setting for model BLOB, this runs on a RPi with a RO filesystem
    nnBlobPath = str((Path(__file__).parent / Path('models/frc2022_openvino_2021.4_5shave_20220118.blob')).resolve().absolute())

    # Create a CameraServer for ShuffleBoard visualization
    cs = CameraServer.getInstance()
    cs.enableLogging()

    # Width and Height have to match Neural Network settings: 300x300 pixels
    width = 300
    height = 300
    output_stream_front_nn = cs.putVideo("FrontNN", width, height)
    if streamDepth:
        output_stream_front_depth = cs.putVideo("FrontDepth", width, height)

    # Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
    pipeline = dai.Pipeline()

    # First, we want the Color camera as the output
#    colorCam = pipeline.createColorCamera()
    colorCam = pipeline.create(dai.node.ColorCamera)
    manip = pipeline.create(dai.node.ImageManip)
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    objectTracker = pipeline.createObjectTracker()

    xoutRgb = pipeline.createXLinkOut()
    if streamDepth:
        xoutDepth = pipeline.createXLinkOut()
    xoutTracker = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    if streamDepth:
        xoutDepth.setStreamName("depth")
    xoutTracker.setStreamName("tracklets")

    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    colorCam.setInterleaved(False)
    colorCam.setIspScale(1,5) # 4056x3040 -> 812x608
    colorCam.setPreviewSize(812, 608)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    colorCam.setFps(25)

    # Use ImageManip to resize to 300x300 with letterboxing: enables a wider FOV
    manip.setMaxOutputFrameSize(270000) # 300x300x3
    manip.initialConfig.setResizeThumbnail(300, 300)
    colorCam.preview.link(manip.inputImage)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(15000)

    objectTracker.setDetectionLabelsToTrack([1, 2, 3, 4])  # track red balls, blue balls, persons and robots
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    manip.out.link(spatialDetectionNetwork.input)
    objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
    objectTracker.out.link(xoutTracker.input)

    if (syncNN):
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    else:
        manip.out.link(xoutRgb.input)

    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # Pipeline is now finished, and we need to find an available device to run our pipeline
    # we are using context manager here that will dispose the device after we stop using it
    #with dai.Device(pipeline, True) as device:
    with dai.Device(pipeline) as device:
        # From this point, the Device will be in "running" mode and will start sending data via XLink

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        if streamDepth:
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        trackletsQueue = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)

        # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
        frame = None
        detections = []

#        startTime = time.monotonic()
#        counter = 0
#        fps = 0
        color = (255, 255, 255)
        image_output_bandwidth_limit_counter = 0

        while True:
            inPreview = previewQueue.get()
            track = trackletsQueue.get()

#            counter+=1
#            current_time = time.monotonic()
#            if (current_time - startTime) > 1 :
#                fps = counter / (current_time - startTime)
#                counter = 0
#                startTime = current_time

            frame = inPreview.getCvFrame()
            if streamDepth:
                depth = depthQueue.get()
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

                ssd=sd.getSubTable(f"FrontCam/Object[{t.id}]")

                ssd.putString("Label", str(label))
                ssd.putString("Status", t.status.name)
    #            sd.putNumber("Confidence", int(detection.confidence * 100))
                ssd.putNumberArray("Location", [int(t.spatialCoordinates.x), int(t.spatialCoordinates.y), int(t.spatialCoordinates.z)])

            # After all the drawing is finished, we show the frame on the screen
            # we can't do that on a headless RPi....           cv2.imshow("preview", frame)
            # Instead publish to CameraServer output stream for NetworkTables or MJPEG http stream\
            # ... and lower the refresh rate to comply with FRC robot wireless bandwidth regulations
            image_output_bandwidth_limit_counter += 1
            if image_output_bandwidth_limit_counter > 1:
                image_output_bandwidth_limit_counter = 0
                output_stream_front_nn.putFrame(frame)

            if streamDepth:
                output_stream_front_depth.putFrame(depthFrameColor)

            # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
            if cv2.waitKey(1) == ord('q'):
                break