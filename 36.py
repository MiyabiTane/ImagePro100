import numpy as np
import cv2

def gray_scale(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def f_C(u):
    if u==0:
        return 1/np.sqrt(2)
    else:
        return 1

def DCT(img,T=8):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=1)
        H,W,C=img.shape

    out=np.zeros((H,W,C),dtype=np.float32)
    #step=H//T

    for i in range(0,H,T):
        for j in range(0,W,T):
            for u in range(T):
                for v in range(T):
                    for x in range(T):
                        for y in range(T):
                            for c in range(C):
                                out[v+j,u+i,c]+=(2*f_C(u)*f_C(v)/T)*img[y+j,x+i,c]*np.cos((2*x+1)*u*np.pi/(2*T))*np.cos((2*y+1)*v*np.pi/(2*T))
    return out


def IDCT(img,T=8,K=8):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=1)
        H,W,C=img.shape
    out=np.zeros((H,W,C),dtype=np.float32)
    #step=H//T
    for i in range(0,H,T):
        for j in range(0,W,T):
            for x in range(T):
                for y in range(T):
                    for u in range(K):
                        for v in range(K):
                            for c in range(C):
                                out[y+j,x+i,c]+=f_C(u)*f_C(v)*img[v+j,u+i,c]*np.cos((2*x+1)*u*np.pi/(2*T))*np.cos((2*y+1)*v*np.pi/(2*T))
    out=out*2/T
    out=np.clip(out,0,255)
    out=out.astype(np.uint8)
    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
#gray_img=gray_scale(img)
img_1=DCT(img)
img_2=IDCT(img_1)
cv2.imwrite("./output_image/output36.jpg",img_2)
cv2.imshow("result",img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
