import time
import numpy as np
import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)
cTime, pTime = 0,0
w, h = 480, 360
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    print(ids)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime= cTime
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    if np.all(ids != None):
        display = aruco.drawDetectedMarkers(frame, corners, ids)
        x1 = (corners[0][0][0][0], corners[0][0][0][1])
        x2 = (corners[0][0][1][0], corners[0][0][1][1])
        x3 = (corners[0][0][2][0], corners[0][0][2][1])
        x4 = (corners[0][0][3][0], corners[0][0][3][1])

        im_dst = frame
        im_src = cv2.imread("Objects/20.jpg")
        size = im_src.shape
        pts_dst = np.array([x1, x2, x3, x4]) #pts1
        pts_src = np.array(
            [
                [0, 0],
                [size[1] - 1, 0],
                [size[1] - 1, size[0] - 1],
                [0, size[0] - 1]
            ], dtype=float
        )

        h, status = cv2.findHomography(pts_src, pts_dst)
        temp = cv2.warpPerspective(
            im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
        im_dst = im_dst + temp
        cv2.imshow('Display', im_dst)
    else:
        display = frame
        cv2.imshow('Display', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()