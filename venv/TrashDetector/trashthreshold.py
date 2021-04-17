import cv2
import numpy as np
cap = cv2.VideoCapture(0)
hsvVals_red = [0, 0, 131, 179, 255, 255]
threshold = 0.2
width, height = 480, 360

def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals_red[0], hsvVals_red[1], hsvVals_red[2]])
    upper = np.array([hsvVals_red[3], hsvVals_red[4], hsvVals_red[5]])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)

    closing = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    # Getting the edge of morphology
    edge = cv2.Canny(closing, 175, 175)
    return mask, edge


def getContours(imgThres, img):
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2
    cv2.drawContours(img, biggest, -1, (255,0,255), 7)
    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)


    cnt = contours[3]
    x1, y1, w1, h1 = cv2.boundingRect(cnt)
    cx1 = x1 + w1 // 2
    cy1 = y1 + h1 // 2
    cv2.drawContours(img, [cnt], 0, (255,0,255), 3)
    cv2.circle(img, (cx1, cy1), 10, (0, 255, 0), cv2.FILLED)

    return cx

while True:
    img = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")
    success, img = img.read()
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)

    imgThres, edge = thresholding(img)

    cx = getContours(imgThres, img)  # for translation
    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    cv2.imshow("edges", edge)
    cv2.waitKey(1)