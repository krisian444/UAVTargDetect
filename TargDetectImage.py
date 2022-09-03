# Module: Target Detection Image
# Description: This script is to take an image and perform target detection. 
#              Contours are drawn on each detected object.
# Name: Krisian Bargas
#       ECE 491 Group 9
# Copyright: yes
# Rev Number: V1
# Rev Notes: n/a

import numpy as np
import cv2
import sys 

print("Image Recognition Start")

# reading / opening image file
file = "fieldpicTest.png"
img = cv2.imread(cv2.samples.findFile(file))

# changing image colorspace to HSV
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# Set range for red color
redRange_Low = np.array([165, 87, 111], np.uint8) 
redRange_Low2 = np.array([0, 200, 120], np.uint8)

redRange_Hi = np.array([180, 255, 255], np.uint8)
redRange_Hi2 = np.array([8, 255, 255], np.uint8)

# create mask for red color
redMask1 = cv2.inRange(hsvImg, redRange_Low, redRange_Hi) 
redMask2 = cv2.inRange(hsvImg, redRange_Low2, redRange_Hi2)

redMask = redMask1 + redMask2

# Set range for yellow color
yellowRange_Low = np.array([24, 160, 180], np.uint8) 
yellowRange_Hi = np.array([34, 255, 255], np.uint8)

# create mask for yellow color
yellowMask = cv2.inRange(hsvImg, yellowRange_Low, yellowRange_Hi) 

# Set range for blue color
blueRange_Low = np.array([100, 100, 100], np.uint8) 
blueRange_Hi = np.array([125, 255, 255], np.uint8)

# create mask for blue
blueMask = cv2.inRange(hsvImg, blueRange_Low, blueRange_Hi)

# Set range for orange
orangeRange_Low = np.array([10, 200, 200])
orangeRange_Hi = np.array([22, 255, 255])

# creat mask for orange
orangeMask = cv2.inRange(hsvImg, orangeRange_Low, orangeRange_Hi)

# Set range for purple
purpleRange_Low = np.array([135, 220, 170])
purpleRange_Hi = np.array([150, 255, 255])

# create mask for purple
purpleMask = cv2.inRange(hsvImg, purpleRange_Low, purpleRange_Hi)

# Morphological Transform, Dilation 
# for each color and bitwise_and operator 
# between imageFrame and mask determines 
# to detect only that particular color 
kernal1 = np.ones((3, 3), "uint8")
kernal2 = np.ones((6, 6), "uint8") 

# For red color
redMask = cv2.erode(redMask, kernal1)
redMask = cv2.dilate(redMask, kernal2)
      
# For yellow color
yellowMask = cv2.erode(yellowMask, kernal1)
yellowMask = cv2.dilate(yellowMask, kernal2)
      
# For blue color
blueMask = cv2.erode(blueMask, kernal1)
blueMask = cv2.dilate(blueMask, kernal2)

# For orange color
orangeMask = cv2.erode(orangeMask, kernal1)
orangeMask = cv2.dilate(orangeMask, kernal2)

# For purple color
purpleMask = cv2.erode(purpleMask, kernal1)
purpleMask = cv2.dilate(purpleMask, kernal2)

# create contour to surround red ==========
contours, hierarchy = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cLen = len(contours)
print("# cont:", cLen)

for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(img, "Red Target", (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,0,255), 6)


# create contour to surround yellow ==========
contours, hierarchy = cv2.findContours(yellowMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(pic)
    if(area > 300):
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 4)
        cv2.putText(img, "Yellow Target", (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,255,255), 6)

# create contour to surround blue ==========
contours, hierarchy = cv2.findContours(blueMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cLen = len(contours)
print("# cont:", cLen)

for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
        cv2.putText(img, "Blue Target", (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,0,0), 6)

# create contour to surround orange ==========
contours, hierarchy = cv2.findContours(orangeMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cLen = len(contours)
print("# cont:", cLen)

for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 4)
        cv2.putText(img, "Orange Target", (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,165,255), 6)

# create contour to surround purple ==========
contours, hierarchy = cv2.findContours(purpleMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cLen = len(contours)
print("# cont:", cLen)

for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if(area > 300):
        x, y, w, h = cv2.boundingRect(contour)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 128), 4)
        cv2.putText(img, "Purple Target", (x,y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (128,0,128), 6)


# show image and write
cv2.imshow("Color Detection", img)

k = cv2.waitKey(0)
if k == ord("s"):
    cv2.imwrite(file + "_colordetected.png", img)
