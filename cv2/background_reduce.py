import cv2
import numpy as np


#====== background reduce by detecting motion
cap = cv2.VideoCapture(0)
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()
