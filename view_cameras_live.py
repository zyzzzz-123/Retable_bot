#!/usr/bin/env python3
"""直接显示摄像头实时画面（独占摄像头，不能与推理同时运行）"""
import cv2
import matplotlib.pyplot as plt

print("打开摄像头...")
cap_front = cv2.VideoCapture('/dev/video0')
cap_wrist = cv2.VideoCapture('/dev/video2')
cap_side = cv2.VideoCapture('/dev/video8')

cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_side.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_side.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("关闭窗口退出（注意：运行此脚本时不能同时运行推理）")

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
ax1.set_title('Front Camera (video0)')
ax2.set_title('Wrist Camera (video2)')
ax3.set_title('Side Camera (video4)')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')

ret1, frame1 = cap_front.read()
ret2, frame2 = cap_wrist.read()
ret3, frame3 = cap_side.read()
im1 = ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) if ret1 else [[0]])
im2 = ax2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) if ret2 else [[0]])
im3 = ax3.imshow(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB) if ret3 else [[0]])

try:
    while plt.fignum_exists(fig.number):
        ret1, frame1 = cap_front.read()
        ret2, frame2 = cap_wrist.read()
        ret3, frame3 = cap_side.read()
        
        if ret1:
            im1.set_data(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        if ret2:
            im2.set_data(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        if ret3:
            im3.set_data(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB))
        
        plt.pause(0.03)
except KeyboardInterrupt:
    pass

cap_front.release()
cap_wrist.release()
cap_side.release()
plt.close()
print("已退出")
