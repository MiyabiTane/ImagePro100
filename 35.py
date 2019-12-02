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

def DFT(img):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #Gを複素行列として定義（そうしないとfor文の中の計算でエラーを吐く）
    #G=img.copy().astype(np.complex)
    G=np.zeros((H,W,C),dtype=np.complex)
    #元画像上の位置
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)
    K=W;L=H
    for c in range(C):
        for l in range(L):
            for k in range(K):
                G[l,k,c]=np.sum(img[:,:,c]*np.exp(-2j*np.pi*(k*x/W+l*y/H)))/np.sqrt(H*W)
    return G

def exchangea_img(img):
    H,W,C=img.shape
    half_H=int(H//2);half_W=int(W//2)
    img_n=img.copy()
    img_n[:half_H,:half_W]=img[half_H:,half_W:]
    img_n[half_H:,:half_W]=img[:half_H,half_W:]
    img_n[half_H:,half_W:]=img[:half_H,:half_W]
    img_n[:half_H,half_W:]=img[half_H:,:half_W]
    return img_n

def BPass_fil(img):
    H,W,C=img.shape
    #画像上の位置
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)
    #中心からの距離
    dis_x=x-W//2
    dis_y=y-H//2
    dist=np.sqrt(dis_x**2+dis_y**2)
    mask=np.zeros((H,W),dtype=np.float32)
    mask[(W//2*0.1)<=dist]=1
    mask[dist>(W//2*0.5)]=0
    #cの分増やす。
    mask=np.repeat(mask,C).reshape(H,W,C)
    img*=mask
    return img


def IDFT(G):
    H,W,C=G.shape
    Img=np.zeros((H,W,C),dtype=np.float32)
    #G上の位置
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)

    for c in range(C):
        for l in range(H):
            for k in range(W):
                Img[l,k,c]=np.abs(np.sum(G[:,:,c]*np.exp(2j*np.pi*(k*x/W+l*y/H))))/np.sqrt(H*W)
    Img=np.clip(Img,0,255)
    Img=Img.astype(np.uint8)

    return Img

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
gray_img=gray_scale(img)
dft_img=DFT(gray_img)
dft_img2=exchangea_img(dft_img)
low_img=BPass_fil(dft_img2)
low_img2=exchangea_img(low_img)
out=IDFT(low_img2)
cv2.imwrite("./output_image/output34.jpg",out)
cv2.imshow("result1",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
