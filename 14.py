import cv2
import numpy as np

def gray_scale(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def bibun(img):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #0パディング
    out=np.zeros((W+2,H+2,C),dtype=np.float)
    out[1:W+1,1:H+1]=img.copy().astype(np.float)
    #縦方向エッジ検出
    K1=[[0.,-1.,0.],[0.,1.,0.],[0.,0.,0.]]
    out_tate=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out_tate[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K1)
    #外れ値を0から255の間に収める。これがないと画像が黒くならない。
    out_tate=np.clip(out_tate,0,255)
    out_tate=out_tate[1:H+1,1:W+1].astype(np.uint8)
    #横方向エッジ検出
    K2=[[0,0,0],[-1,1,0],[0,0,0]]
    out_yoko=out.copy()
    for h in range(1,1+H):
        for w in range(1,1+W):
            for c in range(C):
                out_yoko[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K2)
    out_yoko=np.clip(out_yoko,0,255)
    out_yoko=out_yoko[1:H+1,1:W+1].astype(np.uint8)

    return out_tate,out_yoko

img=cv2.imread("imori.jpg")
gray_img=gray_scale(img)
tate_img,yoko_img=bibun(gray_img)
cv2.imwrite("./output_image/output14_tate.jpg",tate_img)
cv2.imwrite("./output_image/output14_yoko.jpg",yoko_img)
cv2.imshow("result1",tate_img)
cv2.imshow("result2",yoko_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
