import numpy as np
import cv2
import cv2.aruco as aruco
import os
def findarucomarkers(frame, markersize = 4, totalmarkers=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markersize}X{markersize}_{totalmarkers}')
    #key = getattr(aruco, f'DICT_ARUCO_ORIGINAL')
    aruco_dict = aruco.Dictionary_get(key)
    arucoparameters = aruco.DetectorParameters_create()
    corners, ids, rejectedimgpoints = aruco.detectMarkers(gray, aruco_dict, parameters = arucoparameters)
    display = aruco.drawDetectedMarkers(frame, corners, ids)
    return [corners, ids]
def drawaruco(cs, id, frame, ArucoListC, ArucoListArea):
    cx, cy = (cs[0][0][0] +cs[0][2][0]) / 2, (cs[0][0][1] + cs[0][3][1]) / 2
    area = round(cv2.contourArea(cs) ** 0.5)
    ArucoListC.append([cx, cy]), ArucoListArea.append(area)
    cv2.putText(frame, str(int(area)), (int(cx), int(cy)), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 1)
    cv2.putText(frame, str(int(cx)) + str(",") + str(int(cy)), (int(cx), int(cs[0][0][1])), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 1)
    return frame, ArucoListC, ArucoListArea
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        ArucoListArea, ArucoListC, info = [], [], [[0, 0], [0]]
        arucofound = findarucomarkers(frame)
        if len(arucofound[0]) != 0:
            for corners, id in zip(arucofound[0], arucofound[1]):
                frame, info[0], info[1] = drawaruco(corners, id, frame, ArucoListC, ArucoListArea)
                print(info)
        cv2.imshow('Display', frame)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()
