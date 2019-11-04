#そのままの配列で変換すれば良い。
import cv2
import numpy as np
import matplotlib.pyplot as plt

def transform(img):
    img_range=np.ravel(img)
    c=min(img_range)
    d=max(img_range)
    a=0
    b=255

    img=(b-c)*(img-c)/(d-c)+a
    #テキストが間違ってるので注意
    img[np.where(img<a)]=a
    img[np.where(b<=img)]=b

    img=img.astype(np.uint8)

    return img

img=cv2.imread("imori_dark.jpg").astype(np.float)
img_n=transform(img)
plt.hist(np.ravel(img_n),bins=255,rwidth=0.8,range=(0,255))
plt.savefig("./output_image/output21.png")
cv2.imwrite("./output_image/output21.jpg",img_n)
cv2.imshow("result1",img_n)
plt.show()
