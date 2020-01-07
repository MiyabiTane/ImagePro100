import cv2
import numpy as np
import matplotlib.pyplot as plt

def BGR2GRAY(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def gabor(K=111,s=10,g=1.2,l=10,p=0,A=0):
    G=np.zeros((K,K),dtype=np.float32)
    for y in range(K):
        for x in range(K):
            y-=K//2;x-=K//2
            #度→ラジアン
            theta=A*np.pi/180.
            x_d=np.cos(theta)*x+np.sin(theta)*y
            y_d=-np.sin(theta)*x+np.cos(theta)*y
            y+=K//2;x+=K//2
            G[y,x]=np.exp(-(x_d**2+(g**2)*(y_d**2))/(2*(s**2)))*np.cos(2*np.pi*x_d/(l+p))
    #絶対値の和が１になるように正規化
    G/=np.sum(np.abs(G))
    #0~255に正規化
    #G-=np.min(G)
    #G=G*255/np.max(G)
    #G=G.astype(np.uint8)
    return G

def filtering(gray,angle,K=111):
    H,W=gray.shape
    #端の値をコピーしてパディング。上下左右にK_size//2拡張する。
    gray=np.pad(gray,(K//2,K//2),'edge')
    out=np.zeros((H,W),dtype=np.float32)
    G=gabor(K=111,s=1.5,g=1.2,l=3,p=0,A=angle)
    G=G.astype(np.float32)
    for y in range(H):
        for x in range(W):
            out[y,x]=np.sum(gray[y:y+K,x:x+K]*G)
    out=np.clip(out,0,255)
    out=out.astype(np.uint8)
    return out

def sum_flter(gray):
    H,W=gray.shape
    angle=[0,45,90,135]
    result=np.zeros((H,W),dtype=np.float32)
    for i in range(len(angle)):
        out=filtering(gray,angle[i]).astype(np.float32)
        result+=out
    #0~255に正規化。足しちゃってるのでクリップではダメ。
    result=result*255/np.max(result)
    #result=np.clip(result,0,255)
    result=result.astype(np.uint8)
    return result

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
gray=BGR2GRAY(img).astype(np.float32)
out=sum_flter(gray)
cv2.imwrite("./output_image/output80.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
