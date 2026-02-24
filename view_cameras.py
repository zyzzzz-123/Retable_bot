#!/usr/bin/env python3
"""实时显示推理过程中的摄像头画面（从临时文件读取）"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

print("等待推理启动...")
print("关闭窗口或 Ctrl+C 退出")

# 等待文件存在
while not os.path.exists("/tmp/cam_front.jpg"):
    time.sleep(0.5)

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title('Front Camera')
ax2.set_title('Wrist Camera')
ax1.axis('off')
ax2.axis('off')

im1 = ax1.imshow(mpimg.imread("/tmp/cam_front.jpg"))
im2 = ax2.imshow(mpimg.imread("/tmp/cam_wrist.jpg") if os.path.exists("/tmp/cam_wrist.jpg") else mpimg.imread("/tmp/cam_front.jpg"))

try:
    while plt.fignum_exists(fig.number):
        try:
            if os.path.exists("/tmp/cam_front.jpg"):
                im1.set_data(mpimg.imread("/tmp/cam_front.jpg"))
            if os.path.exists("/tmp/cam_wrist.jpg"):
                im2.set_data(mpimg.imread("/tmp/cam_wrist.jpg"))
        except:
            pass
        plt.pause(0.1)
except KeyboardInterrupt:
    pass

plt.close()
print("已退出")
