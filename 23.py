import cv2
import numpy as np
import matplotlib.pyplot as plt

def heitan2(img):
    z_max=np.max(img)
    H,W,C=img.shape
    S=H*W*C

    out=img.copy()
    h=0
    for i in range(256):
        num=np.where(img==i)
        h+=len(img[np.where(img==i)])
        z_new=(z_max/S)*h
        out[num]=z_new
    out=out.astype(np.uint8)

    return out

img=cv2.imread("imori.jpg").astype(np.float)
img_n=heitan2(img)
plt.hist(np.ravel(img_n),bins=255,rwidth=0.8,range=(0,255))
plt.savefig("./output_image/output23.png")
cv2.imwrite("./output_image/output23.jpg",img_n)
cv2.imshow("result1",img_n)
plt.show()
