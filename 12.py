import cv2
import numpy as np

def motion_filter(img):
    W,H,C=img.shape
    #0パディング
    out=np.zeros((W+2,H+2,C),dtype=np.float)
    out[1:W+1,1:H+1]=img.copy()
    #フィルタリング
    K=[[1.0/3,0,0],[0,1.0/3,0],[0,0,1.0/3]]
    out2=out.copy()
    for w in range(1,W+1):
        for h in range(1,H+1):
            for c in range(C):
                out2[w,h,c]=np.sum(out[w-1:w+2,h-1:h+2,c]*K)
    out2=out2[1:W+1,1:H+1].astype(np.uint8)
    return out2

img=cv2.imread("imori.jpg")
out=motion_filter(img)
cv2.imwrite("./output_image/output12.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
