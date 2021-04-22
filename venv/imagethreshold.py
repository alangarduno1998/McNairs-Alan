import cv2
import numpy as np
from scipy import stats
cap = cv2.VideoCapture(r"TrashDetector\VideoofDebris3StutteringFixed.mp4")

def resizeimg(img):
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def otsumethod(img):
    bins_num = 256
    hist, bin_edges= np.histogram(img, bins=bins_num)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def displaythresh(img):
    snrimg = signaltonoise(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret3, thresh3 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    ret4, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
    ret5, thresh5 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    # titles = np.array([['Original Image', 'BINARY'], ['BINARY_INV', 'TRUNC'],[ 'TOZERO', 'TOZERO_INV']],dtype=object)
    images1 = np.concatenate((img, thresh1, thresh2), axis=1)  # concatenate horizontally
    images2 = np.concatenate((thresh3, thresh4, thresh5), axis=1)  # concatenate vertically
    img = np.concatenate((images1, images2), axis=0)
    return img


while True:
    ret, img = cap.read()
    img = cv2.resize(img, (480,360))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgresized = resizeimg(img)
    thold = displaythresh(imgresized)
    cv2.imshow("EH", thold)
    cv2.waitKey(200)