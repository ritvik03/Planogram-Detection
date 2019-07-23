# python dynamic_color_tracking.py --filter HSV --webcam

#import cv2
#import argparse
#import numpy as np
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True,
                    help='Range filter. RGB or HSV')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
    ap.add_argument('-i','--image',required=False, help='put image', action='store_true')
    args = vars(ap.parse_args())

    if not args['filter'].upper() in ['RGB', 'HSV']:
        ap.error("Please speciy a correct filter.")

    return args


def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)
    return values

'''def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global p1,p2

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
        if(len(p1)==0):
            p1 = [(x,y)]
		else:
            p2 = [(x,y)]
            lenthofline =
            cv2.putText(image,"Length of line: "+str(), (int((p1[0]+p2[0])/2)+10,int((p1[0]+p2[0])/2)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)

	# check to see if the left mouse button was released
	cv2.line(image, refPt[0], refPt[1], (255, 255, 255), 2)
	cv2.imshow("Original", image)
'''

def main():
    args = get_arguments()

    range_filter = args['filter'].upper()

    camera = cv2.VideoCapture(0)

    setup_trackbars(range_filter)

    while True:
        if args['webcam']:
            ret, image = camera.read()

            if not ret:
                break

            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            image = cv2.imread("tags.jpg")
            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #x_rect,y_rect,w,h = cv2.boundingRect(c)
            #cv2.rectangle(image,(x_rect,y_rect),(x_rect+w,y_rect+h),(0,255,0),2)
            #box = cv2.minAreaRect(c)
            #box = cv2.BoxPoints(box)
            #box = np.array(box, dtype="int")
            #box = perspective.order_points(box)
            #cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            #cv2.putText(image,"(width: "+str(w)+",height: "+str(h)+")", (x_rect+10,y_rect+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(image, center, 3, (0, 0, 255), -1)
                cv2.putText(image,"centroid", (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
                cv2.putText(image,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)

        # show the frame to our screen
        cv2.imshow("Original", image)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Mask", mask)
        cv2.imwrite("mask.png",mask)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break

        #cv2.setMouseCallback("Original", click)





if __name__ == '__main__':
    main()
