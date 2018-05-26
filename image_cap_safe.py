# file1="C:/Users/Mukundh Bhushan/Desktop/captcha.png"
# file2="C:/Users/Mukundh Bhushan/Desktop/captcha.png"
# file3="C:/Users/Mukundh Bhushan/Desktop/gta.jpg"
import cv2
#import numpy as np
#img=cv2.imread('C:/Users/Mukundh Bhushan/Desktop/1.jpg',cv2.IMREAD_GRAYSCALE)
cap=cv2.VideoCapture(1)

while True:
    _,frame=cap.read()
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    #img=cv2.imread(file1,cv2.IMREAD_GRAYSCALE)
    #img2=cv2.imread(file2,cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (100, 50))
    _, bin = cv2.threshold(img,100,25,1) # inverted threshold (light obj on dark bg)
    bin = cv2.dilate(bin, None)  # fill some holes
    bin = cv2.dilate(bin, None)
    bin = cv2.erode(bin, None)   # dilate made our shape larger, revert that
    bin = cv2.erode(bin, None)
    bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rc = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rc)
    for p in box:
        pt = (p[0],p[1])
        print(pt)
        #cv2.circle(img,pt,5,(200,0,0),2)
    cv2.imshow("plank", img)


    #"""
    ret, thresh = cv2.threshold(img, 10, 255, 12)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.GaussianBlur(img, (3,3), 1)
    #cv2.drawContours(img, contours, -1, (0,255,0), 3)
    epsilon = cv2.arcLength(contours[0],True)
    approx = cv2.approxPolyDP(contours[0],epsilon,True)
    #cv2.imwrite(file2,img)
    cv2.imshow('blur',img)
    #"""
    #edges = cv2.Canny(img,100,200,apertureSize = 3)
    #cv2.imshow('edges',edges)

    #<---canny---->

    low_threshold = 50
    high_threshold = 150
    canny_edges = cv2.Canny(img,low_threshold,high_threshold)
    #cv2.imwrite(file2,canny_edges)
    cv2.imshow('canedg',canny_edges)
    cv2.addWeighted(img, 1,img,1,1)
    cv2.imshow('hough',img)

    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
