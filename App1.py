#23456===============================================================72
import cv2
from ultralytics import YOLO
import numpy as np
#23456===============================================================72
model = YOLO('./best1.pt')
threshold1 = 50 #[threshold for Canny method]
threshold2 = 500 #[threshold for Canny method]
#23456===============================================================72
cap = cv2.VideoCapture(0,700)
#23456===============================================================72
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(frame,conf=0.5,persist=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xx1, yy1, xx2, yy2 = box.xyxy[0]


        img0 = np.array(frame)
        print(img0)
        img1 = img0[int(yy1):int(yy2), int(xx1):int(xx2)]
        print(xx1,xx2,yy1,yy2)

        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3, L2gradient=False)

        cv2.imshow("test",edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elif not ret:
        break

#23456===============================================================72
cap.release()
cv2.destroyAllWindows()