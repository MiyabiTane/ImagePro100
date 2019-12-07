import cv2
import numpy as np

def DCT_w(x, y, u, v,T=8):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2)
    if v == 0:
        cv /= np.sqrt(2)
    theta = np.pi / (2 * T)
    return (( 2 * cu * cv / T) * np.cos((2*x+1)*u*theta) * np.cos((2*y+1)*v*theta))

def BGR_Ycbcr(img):
    H, W, C = img.shape

    ycbcr = np.zeros([H, W, C], dtype=np.float32)

    ycbcr[..., 0] = 0.2990 * img[..., 2] + 0.5870 * img[..., 1] + 0.1140 * img[..., 0]
    ycbcr[..., 1] = -0.1687 * img[..., 2] - 0.3313 * img[..., 1] + 0.5 * img[..., 0] + 128.
    ycbcr[..., 2] = 0.5 * img[..., 2] - 0.4187 * img[..., 1] - 0.0813 * img[..., 0] + 128.

    return ycbcr

def DCT(img,T=8):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=1)
        H,W,C=img.shape

    out=np.zeros((H,W,C),dtype=np.float32)
    #step=H//T
    for c in range(C):
        for i in range(0,H,T):
            for j in range(0,W,T):
                for u in range(T):
                    for v in range(T):
                        for x in range(T):
                            for y in range(T):
                                out[v+i,u+j,c]+=img[y+i,x+j,c]*DCT_w(x,y,u,v)
    return out

def ryosika(dct,T=8):
    H,W,C=dct.shape
    out=np.zeros((H,W,C),dtype=np.float32)

    Q1=np.array([[16, 11, 10, 16, 24, 40, 51, 61],[12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],[24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],[72, 92, 95, 98, 112, 100, 103, 99]],dtype=np.float32)

    Q2=np.array([[17, 18, 24, 47, 99, 99, 99, 99],[18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],[47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],[99, 99, 99, 99, 99, 99, 99, 99]],dtype=np.float32)

    #dct=np.round(dct)

    for y in range(0,H,8):
        for x in range(0,W,8):
            out[y:y+8,x:x+8,0]=np.round(dct[y:y+8,x:x+8,0]/Q1)*Q1
    for y in range(0,H,8):
        for x in range(0,W,8):
            for c in range(1,3):
                out[y:y+8,x:x+8,c]=np.round(dct[y:y+8,x:x+8,c]/Q2)*Q2

    return out

def IDCT(img,T=8,K=8):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=1)
        H,W,C=img.shape
    out=np.zeros((H,W,C),dtype=np.float32)
    #step=H//T
    for c in range(C):
        for i in range(0,H,T):
            for j in range(0,W,T):
                for x in range(T):
                    for y in range(T):
                        for u in range(K):
                            for v in range(K):
                                out[y+i,x+j,c]+=img[v+i,u+j,c]*DCT_w(x,y,u,v)
    out=np.clip(out,0,255)
    out=np.round(out).astype(np.uint8)
    return out

def Ycbcr_BGR(ycbcr):
    H, W, C = ycbcr.shape

    out = np.zeros([H, W, C], dtype=np.float32)
    out[..., 2] = ycbcr[..., 0] + (ycbcr[..., 2] - 128.) * 1.4020
    out[..., 1] = ycbcr[..., 0] - (ycbcr[..., 1] - 128.) * 0.3441 - (ycbcr[..., 2] - 128.) * 0.7139
    out[..., 0] = ycbcr[..., 0] + (ycbcr[..., 1] - 128.) * 1.7718

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
img_2=BGR_Ycbcr(img)
img_3=DCT(img_2)
img_4=ryosika(img_3)
img_5=IDCT(img_4)
out=Ycbcr_BGR(img_5)

cv2.imwrite("./output_image/output40.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
