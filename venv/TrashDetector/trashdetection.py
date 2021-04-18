from djitellopy import tello
import cv2
import numpy as np
#import KeypressModule as kp

frameWidth = 440
frameHeight = 360
width = 440
height = 360

#drone = tello.Tello()
#drone.connect()
#print(drone.get_battery())
#drone.streamon()




def empty(a):
    pass


cv2.namedWindow("HSV")
#cv2.resizeWindow("HSV", 640, 240)
cv2.resizeWindow("HSV", 440, 360)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)
frameCounter = 0

while True:
    cap = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")
    success, img = cap.read()
    #img = drone.get_frame_read().frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    #frame = cv2.flip(frame, 0)
    ret, display = cap.read()
    ret, image = cap.read()
    display = cv2.resize(display, (width, height))
    image = cv2.resize(image, (width, height))
    ret, mask = cap.read()
    mask = cv2.resize(mask, (width, height))
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(g, 60, 60, apertureSize=7, L2gradient=True)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    r, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, h = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(image, contours, -1, (0, 0, 255), thickness=5)
    result = cv2.bitwise_and(mask, mask, mask=t)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    t1 = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    #_,img = cap.read()
    img=result
    img = cv2.resize(img, (frameWidth, frameHeight))
    #img = cv2.flip(img,0)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    print(f'[{h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max}]')

    mask= cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    cv2.imshow('PP STACK', hStack)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()