import cv2
import numpy as np
import glob
width, height = 1280, 720
cap = cv2.VideoCapture("VideoofDebris3StutteringFixed.mp4")
hsvvals_red = [98, 0, 88, 179, 255, 255]
threshold = 0.3
# img_array = []
# out = cv2.VideoWriter('VideoofDebrisdetected.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
def makeframes(cap, width, height):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    image = frame.copy()
    display = frame.copy()
    immask = frame.copy()
    return frame, image, display, immask
def binarythreshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    r, t = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image = cv2.bitwise_and(immask, immask, mask=t)
    return image, gray, t
def colorthreshold(image, hsvvals_red):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvvals_red[0], hsvvals_red[1], hsvvals_red[2]])
    upper = np.array([hsvvals_red[3], hsvvals_red[4], hsvvals_red[5]])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask
def edgethreshold(result, display):
    g = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(g, 60, 60)
    # plotting biggest contour to second frame
    edgeres = cv2.bitwise_and(t, t, mask=edge)
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
    cv2.drawContours(display, contours[0], -1, (0, 0, 255), thickness=5)
    return edgeres
def findcontours(mask, display):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    # cy = y + h // 2
    cv2.putText(display, str("trash"), (cx, y), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 1)
    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # cv2.drawContours(display, biggest, -1, (255, 0, 255), 7)
    return display
while cap.isOpened():
    # -- making frames and resizing
    frame, image, display, immask = makeframes(cap, width, height)

    # -- applying thresholding to image
    image, gray, t = binarythreshold(image)
    #cv2.imshow(" binary thresh ", image)

    # -- applying color thresholding
    result, mask = colorthreshold(image, hsvvals_red)
    #cv2.imshow(" color thresh ", result)

    # -- apply edge thresholding
    edgeresult = edgethreshold(result, display)
    #cv2.imshow(" edge thresh ", edgeresult)

    # -- apply contour to detect objects
    display = findcontours(mask, display)  # color threshold mask and frame to attach contours needs to be predefined
    cv2.imshow(" contours ", display)

    # -- use this if you want to write result to video
    # img_array.append(display)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    #     print(img_array[i])
    # out.release()

    cv2.imshow("Output", np.hstack([display]))
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
