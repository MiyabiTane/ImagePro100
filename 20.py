import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("imori_dark.jpg").astype(np.float)
#ravel()で画素の配列を１次元に変換
#0から255のビンを用意
plt.hist(np.ravel(img),bins=255,range=(0,255))
plt.savefig("./output_image/output20.png")
plt.show()
