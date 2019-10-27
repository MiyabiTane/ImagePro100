import cv2
import numpy as np

def average_filter(img):
    W,H,C=img.shape
    #0パディング
    out=np.zeros((W+2,H+2,C),dtype=np.float)
    out[1:W+1,1:H+1]=img.copy()
    #フィルタリング
    out2=out.copy()
    for w in range(1,W+1):
        for h in range(1,H+1):
            for c in range(C):
                out2[w,h,c]=np.average(out[w-1:w+2,h-1:h+2,c])
    out2=out2[1:W+1,1:H+1].astype(np.uint8)
    return out2

img=cv2.imread("imori.jpg")
out=average_filter(img)
cv2.imwrite("./output_image/output11.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
