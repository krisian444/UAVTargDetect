# Module: Target Detection Live Video
# Description: This script is to take video feed from camera source and perform target detection. 
#              Contours are drawn on each detected object.
# Name: Krisian Bargas
#       ECE 491 Group 9
# Copyright: yes
# Rev Number: V1.3
# Rev Notes: - updated color ranges for mask
# - added the calculation for the size of object area in the image instead of having it as a flat number
# - moved defining color ranges outside of while loop

import numpy as np
import cv2
import sys 

# get webcam / camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# RPi camera is used for target detection and object sizes at a distance will look small
# This part we will calculate the ratio of object size pixels

objSizeReal = 24 # 24 inches in real life target size
FleightHeight = 600 # 50 feet into inches // assumes max camera height to object, can use real time data
focal = 3.04 # 3.04 mm focal length of RPi Camera V2
pixSizeCamera = 1.12*10**-3 # 1.12 micrometers into millimeters

objAreaPix = np.uint16((objSizeReal/FleightHeight * focal) / pixSizeCamera)

# this value was calculated by getting the size:distance ration of target (24 inches:50ft)
# and multiplying with focal length of camera to see how much space it takes in an image
# to translate to pixels, it is divided by the Pixel size given in the RPi camera documentation

# define color ranges
# Set range for red color - 2 ranges for red since it is 180 deg not 360 deg colour space

# Set range for red color
redRange_Low = np.array([165, 87, 111], np.uint8) 
redRange_Low2 = np.array([0, 100, 120], np.uint8)

redRange_Hi = np.array([180, 255, 255], np.uint8)
redRange_Hi2 = np.array([8, 255, 255], np.uint8)

# Set range for yellow color
yellowRange_Low = np.array([24, 120, 150], np.uint8) 
yellowRange_Hi = np.array([34, 255, 255], np.uint8)

# Set range for blue color
blueRange_Low = np.array([100, 100, 100], np.uint8) 
blueRange_Hi = np.array([125, 255, 255], np.uint8)

# Set range for orange
orangeRange_Low = np.array([10, 200, 200])
orangeRange_Hi = np.array([22, 255, 255])

# Set range for purple
purpleRange_Low = np.array([135, 220, 170])
purpleRange_Hi = np.array([150, 255, 255])


# while loop for video stream
while(1):

    ret, img = camera.read()
    
    # break look if video stream not recieved successfully
    if not ret:
        print("Not receiving stream. Exiting")
        break

    # changing image colorspace to HSV
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create mask for red color
    redMask1 = cv2.inRange(hsvImg, redRange_Low, redRange_Hi) 
    redMask2 = cv2.inRange(hsvImg, redRange_Low2, redRange_Hi2)

    redMask = redMask1 + redMask2

    # create mask for yellow color
    yellowMask = cv2.inRange(hsvImg, yellowRange_Low, yellowRange_Hi) 

    # create mask for blue
    blueMask = cv2.inRange(hsvImg, blueRange_Low, blueRange_Hi)

    # create mask for orange color
    orangeMask = cv2.inRange(hsvImg, orangeRange_Low, orangeRange_Hi)

    # create mask for purple
    purpleMask = cv2.inRange(hsvImg, purpleRange_Low, purpleRange_Hi)


    # Morphological Transform, Dilation 
    # for each color to remove noise and retain approx original object
    kernal = np.ones((3, 3), "uint8") 

    # For red color
    redMask = cv2.erode(redMask, kernal)
    redMask = cv2.dilate(redMask, kernal)
        
    # For yellow color
    yellowMask = cv2.erode(yellowMask, kernal)
    yellowMask = cv2.dilate(yellowMask, kernal)
        
    # For blue color
    blueMask = cv2.erode(blueMask, kernal)
    blueMask = cv2.dilate(blueMask, kernal)

    # For orange color
    orangeMask = cv2.erode(orangeMask, kernal)
    orangeMask = cv2.dilate(orangeMask, kernal)

    # For purple color
    purpleMask = cv2.erode(purpleMask, kernal)
    purpleMask = cv2.dilate(purpleMask, kernal)

    # create contour to surround red ================
    contoursR, hierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _ , contour in enumerate(contoursR):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Red Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))

    # create contour to surround yellow ================
    contoursY, hierarchy = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _ , contour in enumerate(contoursY):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, "Yellow Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))


    # create contour to surround blue ===============
    contoursB, hierarchy = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _ , contour in enumerate(contoursB):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Blue Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))

    # create contour to surround orange ==========
    contoursO, hierarchy = cv2.findContours(orangeMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _ , contour in enumerate(contoursO):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(img, "Orange Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255))

    # create contour to surround purple ==========
    contoursP, hierarchy = cv2.findContours(purpleMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _ , contour in enumerate(contoursP):
        area = cv2.contourArea(contour)
        if(area > objAreaPix):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 128), 2)
            cv2.putText(img, "Purple Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128,0,128))

    
    # show video
    cv2.imshow("Color Detection Real Time", img)
    k = cv2.waitKey(1)

    # key press to determine when to stop, write image, close video stream, close video write
    # close all windows
    if k == ord("s"):
        # output last frame shown from webcam / was only used for testing purposes
        #cv2.imwrite("webcamTestimg.png", img)
    

        camera.release()
        cv2.destroyAllWindows()
        break
