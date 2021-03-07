import numpy as np
import cv2
import cv2.aruco as aruco
import os

def loadArucoImages(path):
    ObjectList = os.listdir(path)
    numOfMarkers = len(ObjectList)
    print("Total Number of Objects Detected:", numOfMarkers)
    objDicts = {}
    for imgPath in ObjectList:
        key = int(os.path.splitext(imgPath)[0])
        frameEmbed = cv2.imread(f'{path}/{imgPath}')
        objDicts[key] = frameEmbed
    return objDicts

def FindArucoMarkers(frame, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #key = getattr(aruco, f'DICT_{markersize}X{markerSize}_{totalMarkers}')
    key = getattr(aruco, f'DICT_ARUCO_ORIGINAL')
    aruco_dict = aruco.Dictionary_get(key)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = arucoParameters)
    #print(ids)
    if draw:
        display = aruco.drawDetectedMarkers(frame, corners, ids)
    return [corners, ids]

def EmbedArucoImage(corners, id, frame, frameEmbed, drawId=True):
    p1 = (corners[0][0][0], corners[0][0][1]) #top left corner (x,y)
    p2 = (corners[0][1][0], corners[0][1][1]) #top right corner (x,y)
    p3 = (corners[0][2][0], corners[0][2][1]) #bottom left corner (x,y)
    p4 = (corners[0][3][0], corners[0][3][1]) #bottom right corner (x,y)


    size = frameEmbed.shape # height ,width ,center

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
    frameOut = cv2.warpPerspective(frameEmbed, matrix, (frame.shape[1], frame.shape[0]))
    cv2.fillConvexPoly(frame, pts_dst.astype(int), 0, 16)
    frameOut = frame + frameOut

    #if drawId:
        #cv2.putTex
    return frameOut

def main():
    cap = cv2.VideoCapture(0)
    objDicts = loadArucoImages("Objects")

    while(True):
        ret, frame = cap.read()
        loadArucoImages("Objects")
        ArucoFound = FindArucoMarkers(frame)
        if len(ArucoFound[0])!=0:
            for corners, id in zip(ArucoFound[0], ArucoFound[1]):
                if int(id) in objDicts.keys():
                    frame = EmbedArucoImage(corners, id, frame, objDicts[int(id)])
        cv2.imshow('Display', frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()