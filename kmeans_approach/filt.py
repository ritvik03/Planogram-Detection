import numpy as np
import cv2

img = cv2.imread('extracted3.jpg')
#Z = img.reshape((-1,3))
centroids=img.copy()
#MeanShiftFiltering (Trying)
meanShiftImg = img.copy()
cv2.pyrMeanShiftFiltering(img,20, 45, meanShiftImg,1)
cv2.imshow('Mean Shift',meanShiftImg)
img = meanShiftImg.copy()


Z = img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
#while(K<101):
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS,centroids)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('KMC',res2)
cv2.imshow('centroids',centroids)
print('Centres:')
print(center)

print('Labels:')
print(label)

cv2.waitKey(0)

#for i in range(0,img.shape[0]):
#    for j in range(0,img.shape[1]):
#        k = img[i,j]
#        if(k==centre[0]):
#            color[0].append((i,j))
#        print (k)
