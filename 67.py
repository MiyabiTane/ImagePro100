import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_scale(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    #out=out.astype(np.uint8)
    return out

def hog_step1(img):
    H,W=img.shape
    mag=np.zeros((H,W),dtype=np.float32)
    ang=np.zeros((H,W),dtype=np.float32)
    for y in range(H):
        for x in range(W):
            gx=img[y,min(W-1,x+1)]-img[y,max(0,x-1)]
            #0除算を防ぐため
            if gx==0:
                gx=1e-6
            gy=img[min(H-1,y+1),x]-img[max(0,y-1),x]
            mag[y,x]=np.sqrt(gx**2+gy**2)
            ang[y,x]=np.arctan(gy/gx)
            if ang[y,x]<0:
                ang[y,x]+=np.pi
    #これがないと失敗する。
    ang_n=np.zeros_like(ang,dtype=np.int)
    th_ang=np.pi/9
    for i in range(9):
        ang_n[np.where((th_ang*i<=ang) & (ang<=th_ang*(i+1)))]=i

    return mag,ang_n

def make_hist(ang,N=8):
    H,W=ang.shape
    y_step=H//N
    x_step=W//N
    hist=np.zeros((y_step,x_step,9),dtype=np.float32)
    count=1
    for y in range(y_step):
        for x in range(x_step):
            for j in range(N):
                for i in range(N):
                    print("y,x={},{}".format(y,x))
                    #解答はN->4になっている。わからない。
                    hist[y,x,ang[y*N+j,x*N+i]]+=mag[y*N+j,x*N+i]
    return hist


img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
gray=gray_scale(img)
mag,ang=hog_step1(gray)
hist=make_hist(ang)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(hist[:,:,i])
plt.savefig("./output_image/output67.png")
plt.show()
