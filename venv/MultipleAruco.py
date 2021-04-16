import numpy as np
import cv2
import cv2.aruco as aruco
import os


def loadarucoimages(path):
    objectlist = os.listdir(path)
    numofmarkers = len(objectlist)
    print("Total Number of Objects:", numofmarkers)
    objdicts = {}
    for imgpath in objectlist:
        key = int(os.path.splitext(imgpath)[0])
        frameembed = cv2.imread(f'{path}/{imgpath}')
        objdicts[key] = frameembed
    return objdicts


def findarucomarkers(frame, markersize = 6, totalmarkers=250, draw=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # key = getattr(aruco, f'DICT_{markersize}X{markersize}_{totalmarkers}')
    key = getattr(aruco, f'DICT_APRILTAG_36H11')
    aruco_dict = aruco.Dictionary_get(key)
    arucoparameters = aruco.DetectorParameters_create()
    corners, ids, rejectedimgpoints = aruco.detectMarkers(gray, aruco_dict, parameters = arucoparameters)
    # print(ids)
    if draw:
        display = aruco.drawDetectedMarkers(frame, corners, ids)
    return [corners, ids]

def embedarucoimage(corners, id, frame, frameembed, drawId=True):
    p1 = (corners[0][0][0], corners[0][0][1]) #top left corner (x,y)
    p2 = (corners[0][1][0], corners[0][1][1]) #top right corner (x,y)
    p3 = (corners[0][2][0], corners[0][2][1]) #bottom left corner (x,y)
    p4 = (corners[0][3][0], corners[0][3][1]) #bottom right corner (x,y)


    size = frameembed.shape # height ,width ,center

    pts_dst = np.array([p1, p2, p3, p4])
    #pts_src = np.float32([[0, 0],[size[1] - 1, 0],[size[1] - 1, size[0] - 1],[0, size[0] - 1]])
    pts_src = np.array(
        [
            [0, 0],
            [size[1] - 1, 0],
            [size[1] - 1, size[0] - 1],
            [0, size[0] - 1]
        ], dtype=float
    )
    matrix, status = cv2.findHomography(pts_src,  pts_dst)
    frameout = cv2.warpPerspective(frameembed, matrix, (frame.shape[1], frame.shape[0]))
    cv2.fillConvexPoly(frame, pts_dst.astype(int), 0, 16)
    frameout = frame + frameout

    return frameout

def main():
    cap = cv2.VideoCapture(0)
    objdicts = loadarucoimages("Objects")

    while True:
        ret, frame = cap.read()
        loadarucoimages("Objects")
        arucofound = findarucomarkers(frame)
        if len(arucofound[0]) != 0:
            for corners, id in zip(arucofound[0], arucofound[1]):
                if int(id) in objdicts.keys():
                    frame = embedarucoimage(corners, id, frame, objdicts[int(id)])
        cv2.imshow('Display', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
