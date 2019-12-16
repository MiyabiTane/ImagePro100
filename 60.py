import cv2
import numpy as np

def alphabrend(img1,img2,alpha=0.6):
    H1,W1,C=img1.shape
    H2,W2,C=img2.shape
    H=min(H1,H2)
    W=min(W1,W2)
    img1=cv2.resize(img1,(H,W))
    img2=cv2.resize(img2,(H,W))
    out=img1*alpha+img2*(1-alpha)
    out=out.astype(np.uint8)
    return out

img1=cv2.imread("./input_image/imori.jpg").astype(np.float32)
img2=cv2.imread("./input_image/thorino.jpg").astype(np.float32)
out=alphabrend(img1,img2)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output60.png",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
