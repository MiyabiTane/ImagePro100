import numpy as np
import cv2

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

def make_hist(ang,mag,N=8):
    H,W=ang.shape
    y_step=H//N
    x_step=W//N
    hist=np.zeros((y_step,x_step,9),dtype=np.float32)
    count=1
    for y in range(y_step):
        for x in range(x_step):
            for j in range(N):
                for i in range(N):
                    #print("y,x={},{}".format(y,x))
                    hist[y,x,ang[y*4+j,x*4+i]]+=mag[y*4+j,x*4+i]
    #print("hist shape={}".format(hist.shape))
    return hist

def nor_hist(hist,C=3,epsilon=1):
    H,W,c=hist.shape
    num=C//2
    #hist_n=np.zeros((H,W),dtype=np.float32)
    for y in range(H):
        for x in range(W):
            #問題文から読み取れないけど**2が必要
            hist[y,x]=hist[y,x]/np.sqrt(np.sum(hist[max(y-num,0):min(y+num+1,H),max(x-num,0):min(x+num+1,W)]**2)+epsilon)
    return hist

def GetHog(img):
    gray=gray_scale(img)
    mag,ang=hog_step1(gray)
    hist=make_hist(ang,mag)
    hist_n=nor_hist(hist)
    return hist_n

def FindObjects_Step1(img):
    H,W,C=img.shape
    get_hog=[]

    for y in range(0,H,4):
        for x in range(0,W,4):
            #矩形42*42
            rects1=img[max(0,y-21):min(H+1,y+22),max(0,x-21):min(W+1,x+22)]
            rects1=cv2.resize(rects1,(32,32))
            hog1=GetHog(rects1)
            get_hog.append(hog1.ravel())
            #矩形56*56
            rects2=img[max(0,y-28):min(H+1,y+29),max(0,x-28):min(W+1,x+29)]
            rects2=cv2.resize(rects2,(32,32))
            hog2=GetHog(rects2)
            get_hog.append(hog2.ravel())
            #矩形70*70
            rects3=img[max(0,y-35):min(H+1,y+36),max(0,x-35):min(W+1,x+36)]
            rects3=cv2.resize(rects3,(32,32))
            hog3=GetHog(rects3)
            get_hog.append(hog3.ravel())
    return get_hog

img=cv2.imread("./input_image/imori_1.jpg").astype(np.float32)
get_hog=FindObjects_Step1(img)
print(get_hog)
