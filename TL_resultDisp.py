# This script is to display the X, Y coordinates in image frame of target as part of the target localization program
# Display GPS coordinates of target
# 

import numpy as np
import time
import sys
import cv2

def dispRes(img, X, Y, lat, long):
    #
    X, Y = int(X), int(Y)
    lat, long = float(lat), float(long)
    
    # putting text on img
    cv2.circle(img, (X,Y), 4, (255,255,255), -1)
    cv2.putText(img, "Centroid", (X - 20, Y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, str(X)+',', (X + 30, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, str(Y), (X + 70, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, "Latitude: " + str(lat) +" deg", (X, Y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, "Longitude: " + str(long) + " deg", (X, Y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # displaying text, comment out when not needed
    # cv2.imshow("Info Display", img)
    # k = cv2.waitKey(0)
    # if k == ord("s"):
    #     cv2.imwrite("centroidFinishField.png", img)
    #     #left commented just incase needed to output image
    #     cv2.destroyAllWindows

    return img


# just used for testing, comment out when not needed
def main():

    img = cv2.imread("01.png")
    X, Y = int(1770), int(924)
    lat, long = float(53.63892456), float(-113.2868083)

    res = dispRes(img, X, Y, lat, long)
    cv2.imwrite("DispResult.jpg", res)

if __name__ == '__main__':
    main()