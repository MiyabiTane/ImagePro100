import cv2
import numpy as np

def median_filter(img):
    #0パディング
    H,W,C=img.shape
    out=np.zeros((W+2,H+2,C),dtype=np.float)
    out[1:W+1,1:H+1]=img.copy()
    #フィルタリング
    out2=out.copy()
    for w in range(1,W+1):
        for h in range(1,H+1):
            for c in range(C):
                out2[w,h,c]=np.median(out[w-1:w+2,h-1:h+2,c].reshape(1,-1))
    out2=out2[1:W+1,1:H+1].astype(np.uint8)
    #print(out2)
    return out2

img=cv2.imread("imori_noise.jpg")
out=median_filter(img)
cv2.imwrite("./output_image/output10.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
