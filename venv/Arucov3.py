import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import os
import time
camera_matrix = [[1448, 0, 624], [0, 1448, 316],
                 [0, 0, 1]]
camera_matrix = np.float32(camera_matrix)
dist_coeff = [0.05437520427175414, 0.010684173729094198, 0.003107828628462368, -0.00950183296786585, 4.68352656147056]
dist_coeff = np.float32(dist_coeff)
print("\n camera_matrix: \n" + str(camera_matrix)), print("\n dist_coeff: \n" + str(dist_coeff))
cap = cv2.VideoCapture(0)
cTime, pTime = 0,0
w, h = 480, 360
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters, cameraMatrix=camera_matrix, distCoeff=dist_coeff)
    ArucoListArea, ArucoListC = [], []
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime= cTime
    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    if np.all(ids is not None):
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.02, camera_matrix,
                                                dist_coeff)
        print('\n rvec: \n',rvec, '\n tvec: \n', tvec)
        rows = len(rvec)
        display = aruco.drawDetectedMarkers(frame, corners, ids)
        for var in list(range(rows)):
            aruco.drawAxis(frame, camera_matrix, dist_coeff, rvec[var], tvec[var], 0.02)
            cx = (corners[var][0][0][0] + corners[var][0][3][0]) / 2
            cy = (corners[var][0][0][1] + corners[var][0][3][1]) / 2
            area = cv2.contourArea(corners[var])
            ArucoListArea.append(area)
            ArucoListC.append([cx, cy])
            # cv2.putText(frame, str("area"), (ArucoListC[0], ArucoListC[1]), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 1)
            print(ArucoListC, ArucoListArea)
            im_dst = frame
            drawIm = False # change to True to embed image onto marker
            if drawIm:
                p1 = (corners[0][0][0][0], corners[0][0][0][1])  # top left corner
                p2 = (corners[0][0][1][0], corners[0][0][1][1])  #top right corner
                p3 = (corners[0][0][2][0], corners[0][0][2][1])  # bottom left corner
                p4 = (corners[0][0][3][0], corners[0][0][3][1])  # bottom right corner
                im_src = cv2.imread("Objects/20.jpg")
                size = im_src.shape
                pts_dst = np.array([p1, p2, p3, p4])  # pts1
                pts_src = np.array(
                    [
                        [0, 0],
                        [size[1] - 1, 0],
                        [size[1] - 1, size[0] - 1],
                        [0, size[0] - 1]
                    ], dtype=float
                )
                h, status = cv2.findHomography(pts_src, pts_dst)
                temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
                cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
                im_dst = im_dst + temp
            cv2.imshow('Display', im_dst)
    else:
        display = frame
        cv2.imshow('Display', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
