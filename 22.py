import cv2
import numpy as np
import matplotlib.pyplot as plt

def heitan(img):
    m=np.mean(img)
    s=np.std(img)
    m0=128
    s0=52

    out=img.copy()
    out=(s0/s)*(img-m)+m0

    out=out.astype(np.uint8)

    return out

img=cv2.imread("imori_dark.jpg")
out=heitan(img)
plt.hist(np.ravel(out),bins=255,rwidth=0.8,range=(0,255))
plt.savefig("./output_image/output22.png")
cv2.imwrite("./output_image/output22.jpg",out)
cv2.imshow("result1",out)
plt.show()    
