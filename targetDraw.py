# Module: Draw Targets on Image
# Description: This script is to autogenerate test sets for the target detection script (image version)
# Name: Krisian Bargas
#       ECE 491 Group 9
# Copyright: yes
# Rev Number: 1
# Rev Notes: n/a

from pickletools import uint8
import numpy as np
import cv2

# Draw targets on an existing image
file = "fieldpic.png"
img = cv2.imread(cv2.samples.findFile(file)) 

# get dimensions of image
height, width, channels = img.shape

# defining colors for target using RGB color space
red = [0,0,255]
blue = [252,60,3]
yellow = [3,252,252]
orange = [165,3,252]
purple = [252,3,157]
colorList = np.array([red, blue, yellow, orange, purple], dtype = np.uint8)

# for loop for creating targets for each color in list
for i in range(0, len(colorList)):
    # generate circles between radius of 
    radius = np.random.randint(10, 50)
    # radius = 20
    color = colorList[i].tolist()
    Xcoord = np.random.randint(0, width*0.75)
    Ycoord = np.random.randint(0, height*0.75)
    

    # draw circles
    outputimg = cv2.circle(img, (Xcoord,Ycoord), radius, tuple(color), -1)

# display image
cv2.imshow("Tragets On Field", outputimg)

# save image and write into file, comment out if not needed
k = cv2.waitKey(0)
#if k == ord("s"):
#    cv2.imwrite("fieldpicTest3.png", img)
