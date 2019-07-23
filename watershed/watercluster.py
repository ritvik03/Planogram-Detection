# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
from sklearn.cluster import KMeans


'''def mse(imageA, imageB):
    if(imageA.shape==imageB.shape):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err
    else:
        return 1

def removeImage(array,image):
    for i in range(0,len(array)):
        if mse(array[i],image)==0:
            array.remove(array[i])
    return array'''

def filters(img_rgb,matchesCount,topLeft,bottomRight):
    matchesCountFilter = False
    areaFilter = False
    imageArea = img_rgb.shape[0]*img_rgb.shape[1]
    clusterArea = (bottomRight[1]-topLeft[1])*(bottomRight[0]-topLeft[0])
    areaPercent = 1.0*clusterArea/imageArea
    ########## MATCHES COUNT FILTER ################
    if(matchesCount<4000):
        matchesCountFilter = True

    ######### AREA % FILTER #############
    if(areaPercent<0.7):
        areaFilter = True
        print('area percent: '+str(areaPercent))

    if(matchesCountFilter and areaFilter):
        return True


def findMatchesInOriginal(template,img_rgb):
    w, h, channel = template.shape
    width,height,chan = img_rgb.shape
    originalImage = img_rgb.copy()
    #filteredClusters=[]
    if((width<w) or (height<h)):
        return False
    else:
        res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCORR_NORMED)
        threshold = 0.95
        loc = np.where( res >= threshold)
        topLeft = [width,height]
        bottomRight = [-1,-1]

        matchesCount=0
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (150,0,150), 1)
            matchesCount=matchesCount+1

            if(pt[0]<=topLeft[0]):
                topLeft[0] = pt[0]
            if(pt[1]<=topLeft[1]):
                topLeft[1] = pt[1]
            if(pt[0]>=bottomRight[0]):
                bottomRight[0] = pt[0]+w
            if(pt[1]>=bottomRight[1]):
                bottomRight[1] = pt[1]+h

        print('No. of Matches for this template: '+ str(matchesCount))

        if(filters(img_rgb,matchesCount,topLeft,bottomRight)):
            #####IF FILTER PASSES THEN DO THIS#############
            color = np.random.randint(0,255,(3)).tolist()
            cv2.rectangle(img_rgb,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),(0,0,255),2)
            cv2.imshow('Cluster',img_rgb)
            #cv2.waitKey(0)
            return img_rgb,(topLeft,bottomRight)
        else:
            img_rgb = originalImage.copy()
            return img_rgb,([-1,-1],[-1,-1])

        #return img_rgb

'''def deleteImages(rois,indexDelete):
    finalRois=[]
    for i in indexDelete:
        print(str(i))
        del rois[i]
    return rois
    ########### SECOND APPROACH #####################
    for i in range(0,len(rois)):
        if i not in indexDelete:
            print(str(i))
            finalRois.append(rois[i])
    return finalRois'''

'''def deleteMatching(rois):
    finalRois=rois
    for roiT,roiI in zip(rois,rois):
        if(checkMatch(roiT,roiI)):
            finalRois.remove(roiI)
            #finalRois = removeImage(finalRois,roiI)
            #can I use divide and conquor (merge sort type)
    return finalRois
    ############ SECOND APPROACH #######################
    indexDelete=[]

    for i in range(0,len(rois)):
        for j in range(0,len(rois)):
            if(i!=j):
                if(checkMatch(rois[i],rois[j])):
                    if j not in indexDelete:
                        print('IndexDeleted: '+str(j))
                        indexDelete.append(j)
    finalRois = deleteImages(rois,indexDelete)
    return finalRois'''



def checkMatch(template,img_rgb):
    w, h, channel = template.shape
    width,height,chan = img_rgb.shape
    if((width<w) or (height<h)):
        return False
    else:
        res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCORR_NORMED)
        threshold = 0.95
        loc = np.where( res >= threshold)
        if len(loc) == 0:
            return False
        else:
            return True

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
image = cv2.imread(args["image"])
originalImage = image.copy()
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.imshow("Input", image)

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

i=0
rois=[]
contours=[]
# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	if label == 0:
		continue
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# draw a circle enclosing the object
	x, y, w, h = cv2.boundingRect(c)
	roi = originalImage[y:y+h, x:x+w]
    #rois.append(roi)
    #contours.append(c)
	cv2.imwrite('croppedImages/'+str(label)+'.jpg',roi)
	cv2.rectangle(image, (int(x), int(y)), (int(x+w),int(y+h)), (0, 255, 0), 2)
	cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	rois.append(roi)
	contours.append(c)
	i=i+1

# show the output image
cv2.imshow("Output", image)
cv2.imwrite("results/watershed_"+args["image"],image)
cv2.waitKey(0)

#rois = deleteMatching(rois)
print('No. of ROIS: '+str(len(rois)))

imageReDraw = originalImage.copy()
filteredClusters=[]
clusterProp=[]
for roi in rois:
    image = originalImage.copy()
    matches,(topLeft,bottomRight)=findMatchesInOriginal(roi,image)



    color = np.random.randint(0,255,(3)).tolist()
    if(topLeft[0]>0):
        cv2.rectangle(imageReDraw,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),color,2)
        filteredClusters.append([topLeft,bottomRight])
        centre = [topLeft[0]/2+bottomRight[0]/2,topLeft[1]/2+bottomRight[1]/2]
        #area = abs((bottomRight[0]-topLeft[0])*(bottomRight[1]-topLeft[1]))
        ht = abs(bottomRight[1]-topLeft[1])
        wd = abs(bottomRight[0]-topLeft[0])
        clusterProp.append([centre[0],centre[1],ht,wd])

    #cv2.imshow('Matches',matches)  #### matches contain cluster image of a single ROI
    cv2.imshow('ROI',roi)
    cv2.imshow('All clusters',imageReDraw)
    cv2.waitKey(0)

###### K-Means on basis of cluster location #######
#for X in clusterProp:
#print(p)
print('No. of Total filtered Clusters: '+str(len(filteredClusters)))
# Number of clusters
kmeans = KMeans(n_clusters=int(len(filteredClusters)/6))
# Fitting the input data
kmeans = kmeans.fit(clusterProp)
# Getting the cluster labels
labels = kmeans.predict(clusterProp)
# Centroid values
centroids = kmeans.cluster_centers_
print('Centroid: ')
print(centroids)
i=0
imageKMeans = originalImage.copy()
for c in centroids:
    imageKMeansCurrent = originalImage.copy()
    wd = c[3]
    ht = c[2]
    color = np.random.randint(0,255,(3)).tolist()
    topLeft = [int(c[0]-wd/2),int(c[1]-ht/2)]
    bottomRight = [int(c[0]+wd/2),int(c[1]+ht/2)]
    cv2.rectangle(imageKMeans,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),color,3)
    cv2.rectangle(imageKMeansCurrent,(topLeft[0],topLeft[1]),(bottomRight[0],bottomRight[1]),(0,255,0),3)
    cv2.imshow('KMEANS_CURRENT',imageKMeansCurrent)
    cv2.imwrite('results/KMEANS_Cluster_'+str(i)+'_'+args["image"],imageKMeansCurrent)
    i=i+1
    cv2.waitKey(0)
cv2.imwrite('results/KMEANS_'+args["image"],imageKMeans)
cv2.waitKey(0)
