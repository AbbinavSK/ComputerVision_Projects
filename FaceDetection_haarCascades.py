import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Reading Videos
capture = cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier('Open_CV\haar_face.xml')

while True:
    isTrue, frame  = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Returns the coordinates of the rectangles
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
    
    # Drawing rectangles around the faces
    for x, y, w, h in faces_rect:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)
        cv.putText(frame, 'Face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), thickness=2)

    cv.imshow('Video', frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()