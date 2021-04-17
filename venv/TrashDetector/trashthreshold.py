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
    return mask


def getContours(imgThres, img):
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2
    cv2.drawContours(img, biggest, -1, (255,0,255), 7)
    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    cnt = []

    for i in list(range(len(contours))):
        cnt.append([contours[i]])
        x[i], y[i], w[i], h[i] = cv2.boundingRect(cnt[i])
        cx[i] = x[i] + w[i] // 2
        cy[i] = y[i] + h[i] // 2
        cv2.drawContours(img, cnt[i], -1, (255, 0, 255), 7)
        cv2.circle(img, (cx[i], cy[i]), 10, (0, 255, 0), cv2.FILLED)

    return cx

while True:
    img = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")
    success, img = img.read()
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)

    imgThres = thresholding(img)

    cx, cx[i] = getContours(imgThres, img)  # for translation
    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    cv2.waitKey(1)