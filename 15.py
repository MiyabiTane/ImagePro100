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

def sobel_fil(img):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #0パディング
    out=np.zeros((H+2,W+2,C),dtype=np.float)
    out[1:H+1,1:W+1]=img.copy().astype(np.uint8)
    #縦方向
    K1=[[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]
    out_tate=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out_tate[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K1)
    out_tate=np.clip(out_tate,0,255)
    out_tate=out_tate[1:H+1,1:W+1].astype(np.uint8)
    #横方向
    K2=[[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]
    out_yoko=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out_yoko[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K2)
    out_yoko=np.clip(out_yoko,0,255)
    out_yoko=out_yoko[1:H+1,1:W+1].astype(np.uint8)

    return out_tate,out_yoko

img=cv2.imread("imori.jpg")
gray_img=to_gray(img)
out_tate,out_yoko=sobel_fil(gray_img)
cv2.imwrite("./output_image/output15_tate.jpg",out_tate)
cv2.imwrite("./output_image/output15_yoko.jpg",out_yoko)
cv2.imshow("result1",out_tate)
cv2.imshow("result2",out_yoko)
cv2.waitKey(0)
cv2.destroyAllWindows()
