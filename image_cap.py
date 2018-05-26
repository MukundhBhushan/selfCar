# file1="C:/Users/Mukundh Bhushan/Desktop/captcha.png"
# file2="C:/Users/Mukundh Bhushan/Desktop/captcha.png"
# file3="C:/Users/Mukundh Bhushan/Desktop/gta.jpg"
import cv2
import numpy as np
import imutils
import time

#import numpy as np
#img=cv2.imread('C:/Users/Mukundh Bhushan/Desktop/1.jpg',cv2.IMREAD_GRAYSCALE)
cap=cv2.VideoCapture(1)
#print(cap.get(cv2.CAP_PROP_FPS))
#cap.set(cv2.CAP_PROP_FPS, -1) #cv.CV_CAP_PROP_FPS
#print(cap.get(cv2.CAP_PROP_FPS))
while True:
    _,frame=cap.read()
    #frame=cv2.resize(frame, (300, 300))
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, bin = cv2.threshold(img,100,250,10) # inverted threshold (light obj on dark bg)
    bin = cv2.dilate(bin, None)  # fill some holes
    bin = cv2.dilate(bin, None)
    bin = cv2.erode(bin, None)   # dilate made our shape larger, revert that
    bin = cv2.erode(bin, None)
    bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    #"""
    ret, thresh = cv2.threshold(img, 10, 255, 12)
    #img = cv2.GaussianBlur(img, (3,3), 1)
    #cv2.drawContours(img, contours, -1, (0,255,0), 3)
    epsilon = cv2.arcLength(contours[0],True)
    approx = cv2.approxPolyDP(contours[0],epsilon,True)


    #<---canny---->

    low_threshold = 150
    high_threshold = 250
    canny_edges = cv2.Canny(img,low_threshold,high_threshold)

    try:
        points = cv2.goodFeaturesToTrack(canny_edges, 100, 0.01, 10)
        points = np.int0(points)
        #xo, yo = points[0][0]
        # x1, y1 = points[1][0]
        # x2, y2 = points[2][0]
        # x3, y3 = points[3][0]
        xmin,ymin=300,300
        for point in points:
            x, y = point.ravel()

            # if (x,y)<(xmin,ymin):
            #     xmin,ymin=x,y
            #     cv2.circle(frame, (xmin, ymin), 10, (255, 255, 0), -1)

            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            cnts = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
            cv2.circle(frame, extLeft, 15, (0, 0, 255), -1)
            cv2.circle(frame, extRight, 15, (0, 255, 0), -1)
            cv2.circle(frame, extTop, 15, (255, 0, 0), -1)
            cv2.circle(frame, extBot, 15, (255, 255, 0), -1)

        cv2.imshow('canedg', canny_edges)
        cv2.imshow('frame', frame)



    except:
        _;


    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
