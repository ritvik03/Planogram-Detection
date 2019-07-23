import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

#python featureHomo.py <BiggerImage> <Template>

img_rgb = cv2.imread(sys.argv[1])
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

for r,d,f in os.walk("croppedImages/"):
    for file in f:
        #print(file)
        template = cv2.imread(r+file)
        w, h, channel = template.shape
        width,height,chan = img_rgb.shape
        res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCORR_NORMED)
        threshold = 0.95
        loc = np.where( res >= threshold)

        topLeft = [width,height]
        bottomRight = [-1,-1]

        for pt in zip(*loc[::-1]):
            #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            if(pt[0]<=topLeft[0]):
                topLeft[0] = pt[0]
            if(pt[1]<=topLeft[1]):
                topLeft[1] = pt[1]
            if(pt[0]>=bottomRight[0]):
                bottomRight[0] = pt[0]+w
            if(pt[1]>=bottomRight[1]):
                bottomRight[1] = pt[1]+h

        color = np.random.randint(0,255,(3)).tolist()
        cv2.rectangle(img_rgb,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),color,2)
        cv2.imshow('res',img_rgb)
        cv2.waitKey(0)

cv2.imshow('res',img_rgb)
cv2.waitKey(0)
cv2.imwrite('res.png',img_rgb)
