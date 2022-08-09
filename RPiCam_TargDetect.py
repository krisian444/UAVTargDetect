# Module: Target Detection Live Video with Raspberry Pi Camera
# Description: This script is to take video feed from Raspberry Pi Camera V2 and perform target detection. 
#              Contours are drawn on each detected object.
# Name: Krisian Bargas
#       ECE 491 Group 9
# Copyright: yes
# Rev Number: V1
# Rev Notes: n/a

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

import numpy as np
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    objSizeReal = 24 # 24 inches in real life target size

    altitudeLiDAR = 600 # 50 feet into inches // CHANGE WITH LIDAR READING
    focal = 3.04 # 3.04 mm focal length of RPi Camera V2

    pixSizeCamera = 1.12*10**-3 # 1.12 micrometers into millimeters

    # Find how many pixels the object takes up in the camera frame
    objAreaPix = np.uint16((objSizeReal/altitudeLiDAR * focal) / pixSizeCamera)

    # set ranges for target colors
    # red // two color ranges as in OpenCV bc 180 deg

    redRange_Low = np.array([165, 87, 111], np.uint8) 
    redRange_Low2 = np.array([0, 100, 120], np.uint8)
    redRange_Hi = np.array([180, 255, 255], np.uint8)
    redRange_Hi2 = np.array([8, 255, 255], np.uint8)
    # yellow
    yellowRange_Low = np.array([24, 120, 100], np.uint8) 
    yellowRange_Hi = np.array([34, 255, 255], np.uint8)
    # blue
    blueRange_Low = np.array([100, 100, 150], np.uint8) 
    blueRange_Hi = np.array([125, 255, 255], np.uint8)
    
    # Set range for orange
    orangeRange_Low = np.array([10, 150, 150])
    orangeRange_Hi = np.array([22, 255, 255])
    
    # Set range for purple
    purpleRange_Low = np.array([135, 110, 170])
    purpleRange_Hi = np.array([150, 255, 255])
    
    # create color masks
    
    
    # red
    redMask1 = cv2.inRange(hsvImg, redRange_Low, redRange_Hi) 
    redMask2 = cv2.inRange(hsvImg, redRange_Low2, redRange_Hi2)
    redMask = redMask1 + redMask2
    # yellow
    yellowMask = cv2.inRange(hsvImg, yellowRange_Low, yellowRange_Hi)
    # blue
    blueMask = cv2.inRange(hsvImg, blueRange_Low, blueRange_Hi)

    # create mask for orange color
    orangeMask = cv2.inRange(hsvImg, orangeRange_Low, orangeRange_Hi)

    # create mask for purple
    purpleMask = cv2.inRange(hsvImg, purpleRange_Low, purpleRange_Hi)
    
    # Morphological Transform, Dilation 
    # for each color, remove noise and retain approx original object
    kernal = np.ones((6, 6), "uint8") 
    
    #red
    redMask = cv2.erode(redMask, kernal)
    redMask = cv2.dilate(redMask, kernal)

    #yellow
    yellowMask = cv2.erode(yellowMask, kernal)
    yellowMask = cv2.dilate(yellowMask, kernal)

    #blue
    blueMask = cv2.erode(blueMask, kernal)
    blueMask = cv2.dilate(blueMask, kernal)
    
    # For orange color
    orangeMask = cv2.erode(orangeMask, kernal)
    orangeMask = cv2.dilate(orangeMask, kernal)

    # For purple color
    purpleMask = cv2.erode(purpleMask, kernal)
    purpleMask = cv2.dilate(purpleMask, kernal)

    # create contours to surround masks
    # arbitrary number for area selected, need to calculate more appropriate value for area
    
    # red
    contours, hierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Red Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))

    # yellow
    contours, hierarchy = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(image, "Yellow Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))


    # blue
    contours, hierarchy = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, "Blue Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))

    # create contour to surround orange ==========
    contoursO, hierarchy = cv2.findContours(orangeMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _ , contour in enumerate(contoursO):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(image, "Orange Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255))

    # create contour to surround purple ==========
    contoursP, hierarchy = cv2.findContours(purpleMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _ , contour in enumerate(contoursP):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 128), 2)
            cv2.putText(image, "Purple Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,128))

    
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `s` key was pressed, break from the loop
    if key == ord("s"):
        cv2.destroyAllWindows()
        break
