import cv2
import numpy as np

def expand_i(img,a=1.5):
    H,W,C=img.shape
    out=np.zeros((int(H*a),int(W*a),C),dtype=np.float)

    for h in range(int(H*a)):
        for w in range(int(W*a)):
            for c in range(C):
                #round->四捨五入
                if h==0:
                    out[0,0,c]=img[0,0,c]
                else:
                    out[h,w,c]=img[np.round((h/(a*h)).astype(np.int)),np.round((w,(a*w)).astype(np.int)),c]
    out=out.astype(np.uint8)
    return out

img=cv2.imread("imori.jpg").astype(np.float)
img_n=expand_i(img)
cv2.imwrite("./output_image/output25.jpg",img_n)
cv2.imshow("result1",img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
