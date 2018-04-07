import cv2
import numpy as np


#====== threshold
img = cv2.imread("test_frcnn_00.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# hard threshold
_,threshold_color = cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
_,threshold_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# adaptive threshold
threshold_gray_adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 1)


cv2.imshow("threshold_color", threshold_color)
cv2.imshow("threshold_gray", threshold_gray)
cv2.imshow("threshold_gray_adap", threshold_gray_adap)

while cv2.waitKey(1) & 0xFF != ord('q'):
    pass
cv2.destroyAllWindows()




#====== color filtering
cap = cv2.VideoCapture(0)
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

while cap.isOpened():
    ret, frame = cap.read()

    lower = np.array([0,0,10])
    upper = np.array([255,255,255])
    
    mask = cv2.inRange(frame, lower, upper)
    res = cv2.bitwise_or(frame, frame, mask=mask)

    cv2.imshow('original', frame)
    cv2.imshow('modified', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()




#====== blurring and smoothing
cap = cv2.VideoCapture(0)
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

while cap.isOpened():
    ret, frame = cap.read()

    kernal = np.ones((15,15), np.float32) / 255
    frame_smooth = cv2.filter2D(frame, -1, kernal)
    frame_blur = cv2.GaussianBlur(frame, (15,15), 0)
    frame_median_blur = cv2.medianBlur(frame, 15)

    cv2.imshow('original', frame)
    cv2.imshow('frame_smooth', frame_smooth)
    cv2.imshow('frame_blur', frame_blur)
    cv2.imshow('frame_median_blur', frame_median_blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()
