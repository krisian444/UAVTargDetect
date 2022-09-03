# Module: Target Detection Live Video Multithreaded
# Description: This script is to take video feed from camera source and perform target detection. 
#              Contours are drawn on each detected object.
#              This new version is an attempt to have the process multithreaded in order to optimise RPi resources.
# Name: Krisian Bargas
#       ECE 491 Group 9
# Copyright: yes
# Rev Number: V1.3
# Rev Notes: n/a


import numpy as np
import cv2
import sys 
from threading import Thread
import time

class TargetDetect(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        # checks if video stream open
        while(self.capture.isOpened()):

            (self.status, self.frame) = self.capture.read()
            if not self.status:
                print("Not receiving stream. Exiting")
                break
            time.sleep(.01)

    # function for detecting input color found in current image frame
    def color_detect(self, rangeLow, rangeHi, objArea, color):

        
        # checks which color to be detected by input
        if int(color) == 1:
            inpColor = "Red"
            textColor = [0,0,255]
        elif int(color) == 2:
            inpColor = "Yellow"
            textColor = [3,252,252]
        elif int(color) == 3:
            inpColor = "Blue"
            textColor = [252,50,3]
        elif int(color) == 4: 
            inpColor = "Orange"
            textColor = [165,3,252]
        elif int(color) == 5:
            inpColor = "Purple"
            textColor = [252,3,157]
    

        # changing image colorspace to HSV
        hsvImg = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        # create mask for input colour
        Mask = cv2.inRange(hsvImg, rangeLow, rangeHi) 
        
        # Morphological Transform, Dilation 
        # for each color to remove noise and retain approx original object
        kernal = np.ones((3, 3), "uint8") 

        # For red color
        Mask = cv2.erode(Mask, kernal)
        # redMask = cv2.dilate(redMask, kernal)
            
        # create contour to surround input colour ================
        contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for _ , contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > objArea):
                x, y, w, h = cv2.boundingRect(contour)
                self.frame = cv2.rectangle(self.frame, (x, y), (x + w, y + h), tuple(textColor), 2)
                cv2.putText(self.frame, str(inpColor) + " Target", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tuple(textColor))



    def get_Frame(self):

        return self.frame

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(30)
        if key == ord('s'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == '__main__':

    # RPi camera is used for target detection and object sizes at a distance will look small
    # This part we will calculate the ratio of object size in real life vs how the camera sees it
    # Find how much pixels the object takes up in the image space

    objSizeReal = 24 # 24 inches in real life target size
    FleightHeight = 600 # 50 feet into inches // assumes max camera height to object, can use real time data
    focal = 3.04 # 3.04 mm focal length of RPi Camera V2
    pixSizeCamera = 1.12*10**-3 # 1.12 micrometers into millimeters

    objAreaPix = np.uint16((objSizeReal/FleightHeight * focal) / pixSizeCamera)

    # define color ranges
    # Set range for red color - 2 ranges for red since it is 180 deg not 360 deg colour space
    # In OpenCV
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

    video_stream_widget = TargetDetect()

    while True:
        try:

            # for red upper range
            video_stream_widget.color_detect(redRange_Low, redRange_Hi, objAreaPix, 1)
            # red lower
            video_stream_widget.color_detect(redRange_Low2, redRange_Hi2, objAreaPix, 1)

            # for yellow
            video_stream_widget.color_detect(yellowRange_Low, yellowRange_Hi, objAreaPix, 2)

            # for blue
            video_stream_widget.color_detect(blueRange_Low, blueRange_Hi, objAreaPix, 3)

            # for orange
            video_stream_widget.color_detect(orangeRange_Low, orangeRange_Hi, objAreaPix, 4)

            # for purple
            video_stream_widget.color_detect(purpleRange_Low, purpleRange_Hi, objAreaPix, 5)

            video_stream_widget.show_frame()
            
        except AttributeError:
            pass