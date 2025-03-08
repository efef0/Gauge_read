#23456===============================================================72
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
from datetime import datetime as date
import statistics
import av
import streamlit as st
import pandas as pd
from threading import Event
col_list = ['time','value']
filename = "./Gauge_Read.csv"
#23456===============================================================72
agree = st.checkbox("データをリセットする")
if agree :
    df0 = pd.DataFrame(columns=col_list)
    df0.to_csv(filename, header=col_list, index=False)
# 23456===============================================================72
st.write("閾値1 = 50 , 閾値2 = 500　近辺にて調整")
threshold1 = st.slider("閾値1", max_value=100)  # [threshold for Canny method]
threshold2 = st.slider("閾値2", max_value=1000)  # [threshold for Canny method]
record_interval = st.slider("記録間隔(秒)", min_value=1, max_value=60)  # [capture]
#23456===============================================================72
def callback(frame):
    df = pd.read_csv(filename)

    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3, L2gradient=False)

    height, width, channels = img.shape
#--------------------------------------------------------------------72
    for n in range(10, 300):
        value1 = 0
        lines = cv2.HoughLines(edges, 1, np.pi / 180, n)  # ,minLineLength,maxLineGap)
##-------------------------------------------------------------------72
        if lines is None:
            value1 = 99999
            break
##-------------------------------------------------------------------72
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
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if (x2 - x1) != 0:
                        a0 = (y2 - y1) / (x2 - x1)
                        b0 = y1 - (y2 - y1) / (x2 - x1) * x1
                    else:
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
                break
#--------------------------------------------------------------------72
        elif len(lines) < 2:
            break
#--------------------------------------------------------------------72
    current_time = date.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(img, current_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if value1 != 99999:
        cv2.putText(img, str(int(value1)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        t = date.now()
        tt = date.strftime(t, '%Y-%m-%d %H:%M:%S')
        df_to_add = pd.DataFrame([[tt, int(value1)]], columns=col_list)
        df = pd.concat([df, df_to_add])
        df.to_csv(filename, header=col_list, index=False)
#--------------------------------------------------------------------72
        Event().wait(record_interval)
    return av.VideoFrame.from_ndarray(img,format="bgr24")
# 23456===============================================================72
webrtc_streamer(key="example", video_frame_callback=callback,rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
#23456===============================================================72
ddf = pd.read_csv(filename)
csv = ddf.to_csv(header=col_list, index=False)
st.download_button(label='Download',data=csv,file_name='Gauge_Read.csv')