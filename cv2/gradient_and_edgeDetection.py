import cv2
import numpy as np



#====== gradient
cap = cv2.VideoCapture(0)
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    xx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    yy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

    cv2.imshow('frame', frame)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('xx', xx)
    cv2.imshow('yy', yy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()



#====== edges
cap = cv2.VideoCapture(0)
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

while cap.isOpened():
    ret, frame = cap.read()

    edges = cv2.Canny(frame, 100, 100)

    cv2.imshow('frame', frame)
    cv2.imshow('edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()