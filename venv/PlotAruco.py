import numpy as np
import cv2
import PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from IPython.display import display
import tabulate
from pandas.io.formats.style import Styler

def plot(frame):
    plt.figure()
    plt.imshow(frame)
    plt.show(block=False)


def findarucomarkers(frame, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #key = getattr(aruco, f'DICT_{markersize}X{markerSize}_{totalMarkers}')
    key = getattr(aruco, f'DICT_APRILTAG_36H11')
    aruco_dict = aruco.Dictionary_get(key)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = arucoParameters)
    #print(ids)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    return corners, ids, rejectedImgPoints, frame_markers

def plotid(corners, ids, frame_markers):
    plt.figure()
    plt.imshow(frame_markers)
    for i in range(len(ids)):
        c = corners[i][0]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
    plt.legend()
    plt.show(block=False)

def plotrej(corners,ids,rejectedImgPoints, frame_markers):
    plt.figure()
    plt.imshow(frame_markers, origin="upper")
    if ids is not None:
        for i in range(len(ids)):
            c = corners[i][0]
            plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "+", label="id={0}".format(ids[i]))
    """for points in rejectedImgPoints:
        y = points[:, 0]
        x = points[:, 1]
        plt.plot(x, y, ".m-", linewidth = 1.)"""
    plt.legend()
    plt.show(block=False)

def displaydata(corners):
    corners2 = np.array([c[0] for c in corners])
    data = pd.DataFrame({"x": corners2[:, :, 0].flatten(), "y": corners2[:, :, 1].flatten()},
                    index=pd.MultiIndex.from_product([ids.flatten(),
                                                      ["c{0}".format(i)for i in np.arange(4)+1]], names=["marker", ""]))
    data = data.unstack().swaplevel(0, 1, axis=1).stack()
    data["m1"] = data[["c1", "c2"]].mean(axis=1)
    data["m2"] = data[["c2", "c3"]].mean(axis=1)
    data["m3"] = data[["c3", "c4"]].mean(axis=1)
    data["m4"] = data[["c4", "c1"]].mean(axis=1)
    data["o"] = data[["m1", "m2", "m3", "m4"]].mean(axis=1)
    print(data.to_markdown())

def displayrej(corners):
    corners2 = np.array([r[0] for r in corners])
    data = pd.DataFrame({"x": corners2[:, :, 0].flatten(), "y": corners2[:, :, 1].flatten()},
                    index=pd.MultiIndex.from_product([ids.flatten(),
                                                      ["r{0}".format(i)for i in np.arange(1)+1]], names=["marker", ""]))
    data = data.unstack().swaplevel(0, 1, axis=1).stack()
    data["m1"] = data[["r1"]].mean(axis=1)
    # data["m2"] = data[["c2", "c3"]].mean(axis=1)
    # data["m3"] = data[["c3", "c4"]].mean(axis=1)
    # data["m4"] = data[["c4", "c1"]].mean(axis=1)
    data["o"] = data[["m1"]].mean(axis=1)
    print(data.to_markdown())

def quad_area(data):
    length = data.shape[0]//2
    corners = data[["c1", "c2", "c3", "c4"]].values.reshape(length, 2, 4)
    c1 = corners[:, :, 0]
    c2 = corners[:, :, 1]
    c3 = corners[:, :, 2]
    c4 = corners[:, :, 3]
    e1 = c2-c1
    e2 = c3-c2
    e3 = c4-c3
    e4 = c1-c4
    a = -.5 * (np.cross(-e1, e2, axis=1) + np.cross(-e3, e4, axis=1))
    return a

# -- load image to plot aruco
frame = cv2.imread("results/Tag2_1.jpg")
plot(frame)

# -- find aruco tag and draw markers
corners , ids, rejectedImgPoints, frame_markers = findarucomarkers(frame)
print(corners)
print('\n')
print(rejectedImgPoints)
print('\n')

# -- plot frame with tags and ids
plotid(corners,ids, frame_markers)

# -- plot rejectedImgPoints
plotrej(corners,ids,rejectedImgPoints, frame_markers)

# -- print out results
displaydata(corners)

displayrej(rejectedImgPoints)

# -- this is needed to keep plots up or else it clears when process ends
plt.show()

