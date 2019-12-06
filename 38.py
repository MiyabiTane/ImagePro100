import numpy as np
import cv2

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

def ryosika(dct):
    H,W,C=dct.shape
    step_h=H//8;step_w=W//8
    out=np.zeros((H,W,C),dtype=np.float32)

    Q=np.array([[16, 11, 10, 16, 24, 40, 51, 61],[12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],[24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],[72, 92, 95, 98, 112, 100, 103, 99]],dtype=np.float32)
    dct=np.round(dct)

    for y in range(step_h):
        for x in range(step_w):
            for c in range(C):
                out[8*y:8*(y+1),8*x:8*(x+1),c]=np.round(img[y:y+8,x:x+8,c]/Q)
                out[8*y:8*(y+1),8*x:8*(x+1),c]=out[8*y:8*(y+1),8*x:8*(x+1),c]*Q
    return out

def IDCT(img,T=8,K=4):
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

def PSNR(img1,img2,K=4):
    MSE=0
    #v_max=np.max(img2)
    v_max=255
    print("v_max={}".format(v_max))
    if len(img2.shape)==3:
        H,W,C=img2.shape
    else:
        img2=np.expand_dims(img,axis=1)
        H,W,C=img2.shape

    MSE=np.sum((img1-img2)**2)/(H*W*C) #このままだとcの分も足されてしまう。
    psnr=10*np.log10(v_max**2/MSE)
    bitrate=8*(K**2)/(8**2)
    return psnr,bitrate

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
img_1=DCT(img)
img_2=ryosika(img_1)
img_3=IDCT(img_2)
psnr,bitrate=PSNR(img,img_3)
print("PSRN={},bitrate={}".format(psnr,bitrate))
cv2.imwrite("./output_image/output38.jpg",img_3)
cv2.imshow("result",img_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
