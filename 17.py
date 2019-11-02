import cv2
import numpy as np

def to_gray(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def laplacian_fil(img):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #0パディング
    out=np.zeros((H+2,W+2,C),dtype=np.float)
    out[1:H+1,1:W+1]=img.copy().astype(np.uint8)
    #フィルタリング
    K=[[0,1,0],[1,-4,1],[0,1,0]]
    out2=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out2[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K)
    out2=np.clip(out2,0,255)
    out2=out2[1:H+1,1:W+1].astype(np.uint8)

    return out2

img=cv2.imread("imori.jpg")
gray_img=to_gray(img)
out2=laplacian_fil(gray_img)
cv2.imwrite("./output_image/output17.jpg",out2)
cv2.imshow("result1",out2)
cv2.waitKey(0)
cv2.destroyAllWindows()
