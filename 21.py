import cv2
import numpy as np
import matplotlib.pyplot as plt

def transform(img):
    img_range=np.ravel(img)
    c=min(img_range)
    d=max(img_range)
    a=0
    b=255

    img_range[np.where(img_range<c)]=a
    img_range[np.where((c<=img_range) & (img_range<d))]=(b-c)*(img_range[np.where((c<=img_range) & (img_range<d))]-c)/(d-c)+a
    img_range[np.where(d<=img_range)]=b

    out=img_range.reshape(img.shape)
    return out

img=cv2.imread("imori_dark.jpg").astype(np.float)
img_n=transform(img)
plt.hist(np.ravel(img_n),bins=255,rwidth=0.8,range=(0,255))
plt.savefig("./output_image/output21.png")
cv2.imwrite("./output_image/output21.jpg",img_n)
cv2.imshow("result1",img_n)
plt.show()
