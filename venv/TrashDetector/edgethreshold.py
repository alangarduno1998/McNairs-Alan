import cv2
import numpy as np

width, height = 480, 360
cap = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")

while(cap.isOpened()):
    #reading video to frames
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 0)
    ret, display = cap.read()
    ret, image = cap.read()
    display = cv2.resize(display, (width, height))
    display = cv2.flip(display, 0)
    image = cv2.resize(image, (width, height))
    image = cv2.flip(image, 0)
    #plotting contours to first frame
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(g, 60, 180)

    contours = cv2.findContours(edge,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours[0], -1, (0, 0, 255), thickness=2)
    #plotting biggest contour to second frame
    contours, h = cv2.findContours(edge,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(display, contours[0], -1, (0, 0, 255), thickness=5)

    #plotting contours using binarization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    r, t = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, h = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(image, contours, -1, (0, 0, 255), thickness=5)

    #displaying frames
    #cv2.imshow("Output", frame)

    cv2.imshow("Output", np.hstack([frame,display,image]))
    cv2.imshow('canny', edge)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()