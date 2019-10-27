import cv2
import numpy as np

def gaussian_filter(img,K_size=3,sigma=1.3):
    H,W,C=img.shape
    #0パディング
    #np.pad(out,[(1,1),(1,1)],"constant")
    out=np.zeros((H+2,W+2,C),dtype=np.float)
    out[1:H+1,1:W+1]=img.copy()

    #カーネル
    K=[[1.0/16,2.0/16,1.0/16],[2.0/16,4.0/16,2.0/16],[1.0/16,2.0/16,1.0/16]]
    #フィルタリング
    out2=out.copy()
    for w in range(1,W+1):
        for h in range(H,H+1):
            for c in range(C):
                #outの要素一つ一つとKの掛け算をしている。それらの和。
                out2[w,h,c]=np.sum(out[w-1:w+2,h-1:h+2,c]*K)
                #np.dot(out[w-1:w+2,h-1:h+2,c],K)
    out2=out2[1:H+1,1:W+1].astype(np.uint8)
    return out2

img=cv2.imread("imori_noise.jpg")
out=gaussian_filter(img,K_size=3,sigma=1.3)
cv2.imwrite("./output_image/output9.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
np.pad使い方
>>> a=[[1,2],[3,4]]
>>> import numpy as np
>>> np.pad(a,[(1,1),(1,1)],"constant")
array([[0, 0, 0, 0],
       [0, 1, 2, 0],
       [0, 3, 4, 0],
       [0, 0, 0, 0]])
"""
