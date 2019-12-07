import cv2
import numpy as np

def f_C(u):
    if u==0:
        return 1/np.sqrt(2)
    else:
        return 1

def RGB_Ycbcr(img):
    H,W,C=img.shape
    out=np.zeros((H,W,C),dtype=np.float32)

    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y = 0.299 * R + 0.5870 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    out[:,:,0]=Y
    out[:,:,1]=Cb
    out[:,:,2]=Cr

    return out

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

def ryosika(dct):
    H,W,C=dct.shape
    step_h=H//8;step_w=W//8
    out=np.zeros((H,W,C),dtype=np.float32)

    Q1=np.array([[16, 11, 10, 16, 24, 40, 51, 61],[12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],[24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],[72, 92, 95, 98, 112, 100, 103, 99]],dtype=np.float32)

    Q2=np.array([[17, 18, 24, 47, 99, 99, 99, 99],[18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],[47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99]],dtype=np.float32)

    dct=np.round(dct)

    for y in range(step_h):
        for x in range(step_w):
            out[8*y:8*(y+1),8*x:8*(x+1),0]=np.round(dct[8*y:8*(y+1),8*x:8*(x+1),0]/Q1)*Q1
    for y in range(step_h):
        for x in range(step_w):
            for c in range(1,3):
                out[8*y:8*(y+1),8*x:8*(x+1),c]=np.round(dct[8*y:8*(y+1),8*x:8*(x+1),c]/Q2)*Q2

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

def Ycbcr_RGB(img):
    H,W,C=img.shape
    out=np.zeros((H,W,C),dtype=np.float32)

    Y=img[:,:,0]
    Cb=img[:,:,1]
    Cr=img[:,:,2]

    R = Y + (Cr - 128) * 1.402
    G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    B = Y + (Cb - 128) * 1.7718

    out[:,:,0]=B
    out[:,:,1]=G
    out[:,:,2]=R

    out=np.clip(out,0,255)
    out=out.astype(np.uint8)
    return out


img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
YCbCr=RGB_Ycbcr(img)
YCbCrDCT=DCT(YCbCr)
DCTRyo=ryosika(YCbCrDCT)
RyoIDCT=IDCT(DCTRyo)
out=Ycbcr_RGB(RyoIDCT)

cv2.imwrite("./output_image/output40.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
