import cv2
import numpy as np
import glob
width, height = 480, 360
cap = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")
hsvvals_red = [98, 0, 88, 179, 255, 255]
threshold = 0.3

# img_array = []
# out = cv2.VideoWriter('VideoofDebrisdetected.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)


def makeframes(cap, width, height):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    return frame


def binarythreshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    r, t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    image = cv2.bitwise_and(immask, immask, mask=t)
    return image, gray, t


def colorthreshold(image, hsvvals_red):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvvals_red[0], hsvvals_red[1], hsvvals_red[2]])
    upper = np.array([hsvvals_red[3], hsvvals_red[4], hsvvals_red[5]])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def findcontours(mask, display):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2

    cv2.drawContours(display, biggest, -1, (255, 0, 255), 7)
    cv2.circle(display, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return display


while cap.isOpened():
    # -- making frames and resizing
    frame = makeframes(cap, width, height)
    image = makeframes(cap, width, height)
    display = makeframes(cap, width, height)
    immask = makeframes(cap, width, height)

    # -- applying thresholding to image
    image, gray, t = binarythreshold(image)
    cv2.imshow(" binary thresh ", image)


    # -- applying color thresholding
    #result, mask = colorthreshold(image, hsvvals_red)
    cv2.imshow(" color thresh ", result)

    # -- apply contour to detect objects
    display = findcontours(mask, display)  # color threshold mask and frame to attach contours needs to be predefined
    cv2.imshow(" contours ", display)

    # -- use this if you want to write result to video
    # img_array.append(display)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    #     print(img_array[i])
    # out.release()

    cv2.imshow("Output", np.hstack([frame, image, result, display]))
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
