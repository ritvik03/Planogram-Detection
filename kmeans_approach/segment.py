import cv2
import numpy as np
import math

def find_cnt(image):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray,10,20)
    #cv2.imshow('canny',edges)
    im, contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contoursSorted = sorted(contours, key=cv2.contourArea)
    cv2.imshow('IM',im)
    return im,contoursSorted,hierarchy

def find_if_close(cnt1,cnt2,image):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(0,row1):
        for j in range(0,row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 0.1*max(image.shape[0],image.shape[1]) :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def erodeIt(iterations,image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY)
    kernel = np.ones((9,9),np.uint8)
    erosion = cv2.erode(img_gray,kernel,iterations)
    return erosion

def dilateIt(iterations,image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY)
    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.dilate(img_gray,kernel,iterations)
    return dilation

def mean_shift(img):
    meanShiftImg = img.copy()
    cv2.pyrMeanShiftFiltering(img,20, 45, meanShiftImg,1)
    cv2.imshow('Mean Shift',meanShiftImg)
    return meanShiftImg

def takeArea(box):
    return (box[2]-box[0])*(box[3]-box[1])

class Segment:
    def __init__(self,segments=5):
        #define number of segments, with default 5
        self.segments=segments

    def kmeans(self,image):
        image=cv2.GaussianBlur(image,(7,7),0)
        vectorized=image.reshape(-1,3)
        vectorized=np.float32(vectorized)
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(vectorized,self.segments,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0],image.shape[1])),segmented_image.astype(np.uint8)


    def extractComponent(self,image,label_image,label):
        component=np.zeros(image.shape,np.uint8)
        component[label_image==label]=image[label_image==label]
        return component

    def kmc(self,image):
        Z = image.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        cv2.imshow('KMC',res2)
        return res2,center

if __name__=="__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    ap.add_argument("-n", "--segments", required = False, type = int,
        help = "# of clusters")
    args = vars(ap.parse_args())

    image=cv2.imread(args["image"])
    if len(sys.argv)==3:

        seg = Segment()
        label,result= seg.kmeans(image)
    else:
        seg=Segment(args["segments"])
        label,result=seg.kmeans(image)
    cv2.imshow("segmented",result)
    results=[]
    results.append(seg.extractComponent(image,label,0))
    results.append(seg.extractComponent(image,label,1))
    results.append(seg.extractComponent(image,label,2))
    results.append(seg.extractComponent(image,label,3))
    results.append(seg.extractComponent(image,label,4))

    for i in range(0,args["segments"]):
        cv2.imshow('extracted'+str(i),results[i])
        i=i+1
        cv2.waitKey(0)

    cv2.imwrite("segmented.jpg",result)
    cv2.imwrite("extracted0.jpg",results[0])
    cv2.imwrite("extracted1.jpg",results[1])
    cv2.imwrite("extracted2.jpg",results[2])
    cv2.imwrite("extracted3.jpg",results[3])
    cv2.imwrite("extracted4.jpg",results[4])

    meanshift = mean_shift(results[4])

    cv2.waitKey(0)

    res2,center = seg.kmc(results[4])
    #erodekmc = erodeIt(1,res2)

    erode = erodeIt(1,results[4])
    ret,thresh_erode = cv2.threshold(erode,10,255,cv2.THRESH_BINARY)
    colour_erode = cv2.cvtColor(thresh_erode,cv2.COLOR_GRAY2BGR)
    cv2.imshow('Eroded',colour_erode)

    dilate = dilateIt(1,colour_erode)
    cv2.imshow('Dilated',dilate)
    #cv2.imshow('ErodedKMC',erodekmc)
    ret,thresh_dilate = cv2.threshold(dilate,10,255,cv2.THRESH_BINARY)
    colour_dilate = cv2.cvtColor(thresh_dilate,cv2.COLOR_GRAY2BGR)


    im,contours,heirarchy = find_cnt(colour_dilate)
    #im,contours,heirarchy = find_cnt(colour_erode)
    cv2.waitKey(0)

#    '''
    LENGTH = len(contours)
    print(LENGTH)
    status = np.zeros((LENGTH,1))
    for i,cnt1 in enumerate(contours):
        x = i
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2,image)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    boxUnified = []
    boxes=[]
    maximum = int(status.max())+1
    for i in range(0,maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)

            (x, y, w, h) = cv2.boundingRect(cont)
            boxes.append([x,y, x+w,y+h])

            # need an extra "min/max" for contours outside the frame

            rect = cv2.minAreaRect(cont)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxUnified.append(box)
            unified.append(hull)

    #cv2.drawContours(image,unified,-1,(0,255,0),2)
    cv2.drawContours(image,boxUnified,-1,(255,0,0),2)

    boxes.sort(key=takeArea,reverse=True)

    boxes = np.asarray(boxes)
    left = np.min(boxes[:5,0])
    top = np.min(boxes[:5,1])
    right = np.max(boxes[:5,2])
    bottom = np.max(boxes[:5,3])
    cv2.rectangle(image, (left,top), (right,bottom), (0, 0, 255), 2)

    cv2.imshow('Image',image)
    #cv2.drawContours(thresh,unified,-1,255,-1)

    cv2.waitKey(0)
#    '''
    '''''
    ################################################################## (ADDED NOT USEFUL)

    maxB=max(center[0][0],center[1][0],center[2][0],center[3][0],center[4][0])
    maxG=max(center[0][1],center[1][1],center[2][1],center[3][1],center[4][1])
    maxR=max(center[0][2],center[1][2],center[2][2],center[3][2],center[4][2])

    center = [t for t in center if t.all()!=0]

    minB=min(center[0][0],center[1][0],center[2][0],center[3][0])
    minG=min(center[0][1],center[1][1],center[2][1],center[3][1])
    minR=min(center[0][2],center[1][2],center[2][2],center[3][2])


    for i in center:
        #print('Center '+ str(i)+': ')
        print(i)

    print(maxB,maxG,maxR)
    print(minB,minG,minR)

    imgray = cv2.cvtColor(result4,cv2.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv2.findContours(imgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 3)

    cv2.imshow('Image',image)
    cv2.imshow('Imgray',imgray)

    areaSum=0
    momentSumX=0
    momentSumY=0

    for c in contours:
        M = cv2.moments(c)
        if (M['m00']!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            area = cv2.contourArea(c)
            #momentSumX=+area*cx
            momentSumX=M['m20']*area/M['m00']
            momentSumY=M['m02']*area/M['m00']
            #momentSumY=+area*cy
            areaSum=+area
            #print(cx,cy)
        #print( M )

    centroid = (momentSumX/areaSum,momentSumY/areaSum)
    print(centroid)
    '''''
