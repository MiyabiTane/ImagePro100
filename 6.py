import cv2
import numpy as np

img=cv2.imread("imori.jpg")

img[np.where((0<=img) & (img<64))]=32
img[np.where((64<=img) & (img<128))]=96
img[np.where((128<=img) & (img<192))]=160
img[np.where((192<=img) & (img<256))]=224

cv2.imwrite("./output_image/output6.jpg",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
