import cv2
import numpy as np
import scipy.io
import scipy.stats
frameWidth = 480
frameHeight = 360
hsvvals_red = [98, 0, 0, 179, 255, 255]
def blobdet(img, blob=False):
    if blob:
        detector = cv2.SimpleBlobDetector()
        params = cv2.SimpleBlobDetector_Params()
        #params.minThreshold = 0;
        #params.maxThreshold = 150;
        params.filterByCircularity = True
        params.minCircularity = 0.6

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        blobb = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('blobb', blobb)
        return blobb
    else:
        return img
def colorthreshold(image, hsvvals_red, colort=False):
    if colort:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([hsvvals_red[0], hsvvals_red[1], hsvvals_red[2]])
        upper = np.array([hsvvals_red[3], hsvvals_red[4], hsvvals_red[5]])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    else:
        return image
def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))
def edgethreshold(result, display, edget=False):
    if edget:
        g = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(g, 560, 560)
        # plotting biggest contour to second frame
        edgeres = cv2.bitwise_and(result, result, mask=edge)
        contours, h = cv2.findContours(edge,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(display, contours[0], -1, (0, 0, 255), thickness=2)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        # cy = y + h // 2
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 255), 3)
        cv2.putText(display, str("trash"), (cx, y), cv2.FONT_ITALIC, 0.7, (255, 0, 255), 1)
        # cv2.drawContours(display, contours[0], -1, (0, 0, 255), thickness=5)
        cv2.imshow('edgeres', edgeres)
        return edgeres
    else:
        return result
def preprocessing(img,prep=False):
    if prep:

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        r, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        result = cv2.bitwise_and(img, img, mask=t)
        cv2.imshow('preprocessing', result)
        cv2.imshow('t', t)
        return result
    else:
        return img
def filtering(img, blur=False):
    if blur:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow('gaussianBlur', blur)
        return blur
    else:
        return img

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

cap = cv2.VideoCapture(r"C:\Users\alang\PycharmProjects\McNair\venv\TrashDetector\VideoofDebris3StutteringFixed.mp4")
frameCounter = 0

while True:
    _, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    original = img

    # -- change to True for filtering
    img=filtering(img, blur=True)

    # -- calculate snr
    snr = calculate_psnr(original,img)
    print(snr)

    # -- histogram equalization
    equ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(equ)
    cv2.imshow('hist', equ)
    #img=equ

    dft = cv2.dft(np.float32(equ), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = equ.shape
    #rows, cols = int(rows), int(cols)
    crow, ccol = round(rows / 2), round(cols / 2)
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.imshow('fft',img_back)

    # -- change to True for preprocessing
    img = preprocessing(img, prep=True)

    # -- change to True for preprocessing
    img = colorthreshold(img, hsvvals_red, colort=True)

    # -- switch to true for edgethresholding
    img = edgethreshold(img, original, edget=True)

    # -- change to true for blob detection
    img = blobdet(img, blob=False)

    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    #print(f'[{h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max}]')

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    cv2.imshow('Processing Stack', hStack)
    cv2.imshow('Original', original)
    if cv2.waitKey(100) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()