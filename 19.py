import cv2
import numpy as np
import math

def to_gray(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def log_fil(img,sigma=3,size=5):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #0パディング
    pad=size//2
    out=np.zeros((H+pad*2,W+pad*2,C),dtype=np.float)
    out[pad:H+pad,pad:W+pad]=img.copy().astype(np.uint8)
    #Logフィルタを求める
    K=np.zeros((size,size),dtype=np.float)
    #-2<=x<=2,-2<=y<=2
    for x in range(-pad,-pad+size):
        for y in range(-pad,-pad+size):
            K[x+pad,y+pad]=((x**2+y**2-(sigma**2))*math.exp(-(x**2+y**2)/(2*(sigma**2))))
    K/=(2*math.pi*(sigma**6))
    #書いてないけどこれが必要
    K/=K.sum()
    #テスト
    print(K)
    #フィルタリング
    out2=out.copy()
    for h in range(pad,H+pad):
        for w in range(pad,W+pad):
            for c in range(C):
                out2[h,w,c]=np.sum(out[h-pad:h+pad+1,w-pad:w+pad+1,c]*K)
    out2=np.clip(out2,0,255)
    out2=out2[pad:H+pad,pad:W+pad].astype(np.uint8)

    return out2

img=cv2.imread("imori_noise.jpg")
gray_img=to_gray(img)
out2=log_fil(gray_img)
cv2.imwrite("./output_image/output19.jpg",out2)
cv2.imshow("result1",out2)
cv2.waitKey(0)
cv2.destroyAllWindows()
