import numpy as np
import cv2
import cv2.aruco as aruco
import time
cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
cap.set(cv2.CAP_PROP_FPS,60)
pTime, cTime = 0, 0
w, h = 480, 360
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
    frame = aruco.drawDetectedMarkers(frame, corners)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime= cTime
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Display', frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()