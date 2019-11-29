import cv2
import numpy as np

def DFT(img):
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
    #G=abs(G)
    #G=G*255/G.max()
    #G=G.astype(np.uint8)
    return G

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
img_1=DFT(img)
img_2=IDFT(img_1)
G=np.abs(img_1)*255/np.abs(img_1).max()
G=G.astype(np.uint8)
cv2.imwrite("./output_image/output32_1.jpg",G)
cv2.imwrite("./output_image/output32_2.jpg",img_2)
cv2.imshow("result1",G)
cv2.imshow("result2",img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
