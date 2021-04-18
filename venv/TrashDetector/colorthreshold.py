import cv2
import numpy as np

width, height = 480, 360
cap = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")
hsvVals_red = [0, 0, 131, 179, 255, 255]
threshold = 0.3

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
    #applying color thresholding
    hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals_red[0], hsvVals_red[1], hsvVals_red[2]])
    upper = np.array([hsvVals_red[3], hsvVals_red[4], hsvVals_red[5]])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(display, display, mask=mask)
    #apply contour to detect objects
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2
    cv2.drawContours(display, biggest, -1, (255,0,255), 7)
    cv2.circle(display, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    #plotting contours using binarization


    #displaying frames
    #cv2.imshow("Output", frame)

    cv2.imshow("Output", np.hstack([frame,result,display]))
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()