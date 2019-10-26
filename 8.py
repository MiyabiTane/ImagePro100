import cv2
import numpy as np

def max_p(img):
    out=img.copy()
    H,W,C=out.shape
    for i in range(16):
        for j in range(16):
            for c in range(C):
                out[8*i:8*(i+1),8*j:8*(j+1),c]=np.max(out[8*i:8*(i+1),8*j:8*(j+1),c]).astype(np.uint8)
    return out

img=cv2.imread("imori.jpg")
out=max_p(img)
cv2.imwrite("./output_image/output8.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
