import numpy as np
import cv2
import cv2.aruco as aruco
import os
camera_matrix = np.float32([[1448, 0, 624], [0, 1448, 316], [0, 0, 1]])
dist_coeff = np.float32([0.05437520427175414, 0.010684173729094198, 0.003107828628462368, -0.00950183296786585, 4.68352656147056])
def findarucomarkers(frame, markersize = 4, totalmarkers=50, draw=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markersize}X{markersize}_{totalmarkers}')
    #key = getattr(aruco, f'DICT_APRILTAG_36H11')
    aruco_dict = aruco.Dictionary_get(key)
    arucoparameters = aruco.DetectorParameters_create()
    corners, ids, rejectedimgpoints = aruco.detectMarkers(gray, aruco_dict, parameters = arucoparameters, cameraMatrix=camera_matrix, distCoeff=dist_coeff)
    if draw:
        display = aruco.drawDetectedMarkers(frame, corners, ids)
    return [corners, ids]
def findaruco(cs, id, frame, AListC, AListA,AListR, AListT , drawPose=True):
    cx, cy = (cs[0][0][0] +cs[0][2][0]) / 2, (cs[0][0][1] + cs[0][3][1]) / 2
    area = (round(cv2.contourArea(cs) ** 0.5))
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(cs, 0.02, camera_matrix, dist_coeff)
    AListC.append((cx, cy)), AListR.append(rvec), AListT.append(tvec), AListA.append(area)
    if drawPose:
        aruco.drawAxis(frame, camera_matrix, dist_coeff, rvec, tvec, 0.1)
    return frame, AListC, AListA, AListR, AListT
def main():
    cap = cv2.VideoCapture(0)
    AListA, AListC, AListR, AListT = [], [], [], []
    while True:
        info = [[0, 0], [0], [0, 0, 0], [0, 0, 0]]
        ret, frame = cap.read()
        arucofound = findarucomarkers(frame)
        if len(arucofound[0]) != 0:
            for corners, id in zip(arucofound[0], arucofound[1]):
                frame, info[0], info[1][0], info[2], info[3] = findaruco(corners, id, frame, AListC, AListA, AListR, AListT, drawPose=True)
        print(info)
        cv2.imshow('Display', frame)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()