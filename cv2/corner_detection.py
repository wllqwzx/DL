import cv2
import numpy as np



#====== corner detection
cap = cv2.VideoCapture(0)
print("original width:", cap.get(3))
print("priginal height:", cap.get(4))
cap.set(3, 640) # set video height to 640
cap.set(4, 360) # set video width to 360

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 20000, 0.05, 10)
    corners = np.int32(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(frame, (x,y), 3, 255, -1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()
