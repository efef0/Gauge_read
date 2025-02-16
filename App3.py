#23456===============================================================72
import cv2
from ultralytics import YOLO
#import time
import numpy as np
from datetime import datetime as date
#import pandas as pd
import streamlit as st
import statistics
#23456===============================================================72
model = YOLO('./best1.pt')
threshold1 = 50 #[threshold for Canny method]
threshold2 = 500 #[threshold for Canny method]
minLineLength = 100
maxLineGap = 10
#23456===============================================================72
st.markdown('Analog Gauge Reading App')
camera_index = st.number_input('input your camera device number',value=0,step=1)
camera_index = int(camera_index)
placeholder = st.empty()
#23456===============================================================72
cap = cv2.VideoCapture(camera_index,700)
#23456===============================================================72
stop_button_pressed = st.button("Stop")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
#        print(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(frame,conf=0.5,persist=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                xx1, yy1, xx2, yy2 = box.xyxy[0]

        img0 = np.array(frame)
        img1 = img0[int(yy1):int(yy2), int(xx1):int(xx2)]

        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3, L2gradient=False)

        height, width, channels = img1.shape

        for n in range(10, 300):
            value1 = 0
            lines = cv2.HoughLines(edges, 1, np.pi / 180, n)  # ,minLineLength,maxLineGap)
            if lines is None:
                value1 = 99999
#                print("I can't find 2 lines")
                break

            elif len(lines) == 2:
                aa = []
                bb = []
                theta_t = []

                for m in range(len(lines)):
                    for rho, theta in lines[m]:
                        theta_t.append(theta)
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        if (x2 - x1) != 0:
                            a0 = (y2 - y1) / (x2 - x1)
                            b0 = y1 - (y2 - y1) / (x2 - x1) * x1
                        else:
#                            print("I can't find 2 lines")
                            break

                        aa.append(a0)
                        bb.append(b0)

                if len(aa) >= 2:
                    if (aa[0] - aa[1]) != 0:

                        x_t = (bb[1] - bb[0]) / (aa[0] - aa[1])
                        y_t = (aa[0] * bb[1] - bb[0] * aa[1]) / (aa[0] - aa[1])

                        if x_t < width / 2:
                            theta_hor = statistics.mean(theta_t) * 180 / np.pi
#                            value1 = 50 + (50 / 180) * theta_hor
                            value1 = theta_hor
                        else:
                            theta_hor = 270 - (90 - statistics.mean(theta_t) * 180 / np.pi)
#                            value1 = 0 + (50 / 180) * (theta_hor - 180)
                            value1 = theta_hor
                    break

                else:
#                    print("I can't find 2 lines")
                    break

            elif len(lines) < 2:
                print("I can't find 2 lines")

# 23456===============================================================72
        current_time = date.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img1, current_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if value1 != 99999:
            cv2.putText(img1, str(int(value1)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        placeholder.image(img1)




        if stop_button_pressed:
            break



#23456===============================================================72
cap.release()
cv2.destroyAllWindows()