import numpy as np
import cv2

img = cv2.imread('shelf1.jpg')
Z = img.reshape((-1,3))

#MeanShiftFiltering (Trying)
meanShiftImg = img.copy()
cv2.pyrMeanShiftFiltering(img,20, 45, meanShiftImg,1)
cv2.imshow('Mean Shift',meanShiftImg)
img = meanShiftImg.copy()
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
#while(K<101):
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
print(str(K))
print('Centre:')
i=0
while(i<K):
    print(center[i])
    i=i+1

#print('Label:')
#print(label)

#point = tuple(map(int, center))
color=(255,0,0)
#cv2.circle(res2,center[0],3,color,-1,8)
kernel = np.ones((5,5),np.uint8)
blur2 = cv2.GaussianBlur(res2,(5,5),0)
blur = cv2.GaussianBlur(blur2,(5,5),0)
cv2.imshow('Blurred',blur)
erosion1 = cv2.erode(blur,kernel,iterations = 1)
erosion2 = cv2.erode(img,kernel,iterations = 1)
cv2.imshow('Erosion New',erosion2)
#erosion2 = cv2.erode(blur,kernel,iterations = 2)
#dilation= cv2.dilate(res2,kernel,iterations = 1)
#cv2.imshow('Eroded1',erosion1)
#cv2.imshow('Eroded2',erosion2)
#cv2.imshow('Dilation',dilation)

#cv2.imshow('res2',res2)
#cv2.imwrite('TestK/'+str(K)+'.jpg',res2)
#print(str(K))
#K=K+1

#################################CONTOURS#########################################

imgray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',imgray)
edges = cv2.Canny(imgray,10,20)
cv2.imshow('canny',edges)
kernel2 = np.ones((2,2),np.uint8)
dilateEdges = cv2.dilate(edges,kernel2,iterations = 1)
cv2.imshow('dilateEdges',dilateEdges)
im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im3, contours2, hierarchy2 = cv2.findContours(dilateEdges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,0), 1)
contoursSorted = sorted(contours, key=cv2.contourArea)
contoursSorted2 = sorted(contours2, key=cv2.contourArea)
i=0
while(i<=5):
    x,y,w,h = cv2.boundingRect(contoursSorted[-i])
    print(x,y,w,h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3,8,0)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    x,y,w,h = cv2.boundingRect(contoursSorted2[-i])
    print(x,y,w,h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3,8,0)


    i=i+1

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
