import cv2
import numpy as np

width, height = 480, 360
cap = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")

while(cap.isOpened()):
    #reading video to frames
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    #frame = cv2.flip(frame, 0)
    ret, display = cap.read()
    ret, image = cap.read()
    display = cv2.resize(display, (width, height))
    #display = cv2.flip(display, 0)
    image = cv2.resize(image, (width, height))
    #image = cv2.flip(image, 0)

    ret, mask = cap.read()
    mask = cv2.resize(mask, (width, height))
    #plotting contours to first frame

    #applying binary threshold
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    r, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_TOZERO)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # -- applying canny edge detection
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(g, 150, 150, L2gradient=True)
    # plotting contours to display frame
    contours, h = cv2.findContours(edge,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(display, contours[0], -1, (0, 0, 255), thickness=5)
    result = cv2.bitwise_and(t, t, mask=edge)
    cv2.imshow("Output", np.hstack([frame,display,image]))
    cv2.imshow('canny edges', edge)
    cv2.imshow('binary threshold', t)
    cv2.imshow('canny threshold', result)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()