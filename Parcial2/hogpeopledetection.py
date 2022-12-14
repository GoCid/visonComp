import cv2
from cv2 import HOGDescriptor
import imutils


hog = HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
            padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25)& 0xFF == ord('q'):
            break
    else:
        break