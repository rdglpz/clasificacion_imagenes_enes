# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

'''
Referencias:

https://www.life2coding.com/crop-image-using-mouse-click-movement-python/
'''

import cv2

#import numpy as np
#from PIL import Image

cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = cv2.imread('mapa.png')

oriImage = image.copy()
global n
n = 0
def mouse_crop(event, x, y, flags, param):
    global n
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
#            image = PIL.Image.fromarray(roi, "RGB")
            cv2.imwrite("out_2"+str(n)+".jpg", roi)
            n +=1

            
            

            
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
f = True
while f:
    i = image.copy()
    if not cropping:
        cv2.imshow("image", image)

        
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)
#    cv2.waitKey(0)
#    cv2.imshow("image", image)
    k = cv2.waitKey(0) & 0xFF
    print(k)
    if k == 27:
        f = False
#        cv2.destroyAllWindows()
#        break

    
    
# close all open windows
cv2.destroyAllWindows()


p3GL7a1g
