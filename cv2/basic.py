import cv2
import numpy as np
import matplotlib.pyplot as plt


#====== read, show and write a image
img = cv2.imread("test_frcnn_00.jpg", cv2.IMREAD_GRAYSCALE) # note cv2 read color pic in BGR order, not RGB
#cv2.IMREAD_COLOR : default: Loads a color image. Any transparency of image will be neglected.
#cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
#cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel

cv2.imshow("title", img)
while cv2.waitKey(1) & 0xFF != ord('q'):
    pass
cv2.destroyAllWindows()

# cv2.imwrite('new.npg', img)



#====== Capture Video from Camera:
cap = cv2.VideoCapture(0)   # 0 is the first camera, 1 is the second, ...
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # show two windows: original and modified
    cv2.imshow('original', frame)
    cv2.imshow('modified', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()



#====== Saving a Video
cap = cv2.VideoCapture(0)
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    # write the flipped frame
    out.write(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
out.release()
cap.release()
cv2.destroyAllWindows()



#====== Playing Video from file
cap = cv2.VideoCapture('output.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()