import cv2
import numpy as np

#４近傍
def check1(x,y,image):
    H,W=image.shape
    c1=image[max(0,y-1),x] #上
    c2=image[y,max(0,x-1)] #左
    c3=image[y,min(x+1,W-1)] #右
    c4=image[min(y+1,H-1),x] #下
    if c1==0 or c2==0 or c3==0 or c4==0:
        return 1
    return 0

#８ー連結数
def check2(x,y,rem):
    H,W=rem.shape
    x4=np.abs(1-int(rem[max(y-1,0),max(x-1,0)]))
    x3=np.abs(1-int(rem[max(y-1,0),x]))
    x2=np.abs(1-int(rem[max(0,y-1),min(x+1,W-1)]))
    x5=np.abs(1-int(rem[y,max(x-1,0)]))
    x1=np.abs(1-int(rem[y,min(x+1,W-1)]))
    x6=np.abs(1-int(rem[min(y+1,H-1),max(0,x-1)]))
    x7=np.abs(1-int(rem[min(y+1,H-1),x]))
    x8=np.abs(1-int(rem[min(y+1,H-1),min(x+1,W-1)]))

    S=int((x1-x1*x2*x3)+(x3-x3*x4*x5)+(x5-x5*x6*x7)+(x7-x7*x8*x1))
    if S==1:
        return 1
    return 0

#絶対値の合計
def check3(x,y,rem):
    H,W=rem.shape
    x4=np.abs(rem[max(y-1,0),max(x-1,0)])
    x3=np.abs(rem[max(y-1,0),x])
    x2=np.abs(rem[max(0,y-1),min(x+1,W-1)])
    x5=np.abs(rem[y,max(x-1,0)])
    x1=np.abs(rem[y,min(x+1,W-1)])
    x6=np.abs(rem[min(y+1,H-1),max(0,x-1)])
    x7=np.abs(rem[min(y+1,H-1),x])
    x8=np.abs(rem[min(y+1,H-1),min(x+1,W-1)])

    if x1+x2+x3+x4+x5+x6+x7+x8>=2:
        return 1
    return 0

#８近傍
def check4(x,y,label):
    H,W=label.shape
    c1=label[max(0,y-1),x] #上
    c2=label[y,max(0,x-1)] #左
    c3=label[max(0,y-1),max(0,x-1)]#左上
    c4=label[max(0,y-1),min(x+1,W-1)]#右上
    c5=label[y,min(x+1,W-1)] #右
    c6=label[min(y+1,H-1),x] #下
    c7=label[min(y+1,H-1),max(0,x-1)] #左下
    c8=label[min(H-1,y+1),min(W-1,x+1)] #右下

    list=[c1,c2,c3,c4,c5,c6,c7,c8]
    if 1 in list:
        return 1
    return 0

#xnについて
def check5(x,y,rem):
    H,W=rem.shape
    x4=rem[max(y-1,0),max(x-1,0)]
    x3=rem[max(y-1,0),x]
    x2=rem[max(0,y-1),min(x+1,W-1)]
    x5=rem[y,max(x-1,0)]
    x1=rem[y,min(x+1,W-1)]
    x6=rem[min(y+1,H-1),max(0,x-1)]
    x7=rem[min(y+1,H-1),x]
    x8=rem[min(y+1,H-1),min(x+1,W-1)]
    list=[x1,x2,x3,x4,x5,x6,x7,x8]

    for i in range(len(list)):
        if list[i]!=-1:
            return 1
        else:
            keep=list[i]
            list[i]=0
            S=int((list[0]-list[0]*list[1]*list[2])+(list[2]-list[2]*list[3]*list[4])+(list[4]-list[4]*list[5]*list[6])+(list[6]-list[6]*list[7]*list[0]))
            if S==1:
                return 1
            list[i]=keep
    return 0


def scanning(img):
    H,W,C=img.shape
    rem=np.zeros((H,W),dtype=np.int) #black
    rem[img[:,:,0]>0]=1 #white

    while True:
        flag=0
        #tmpを使わないと更新途中の画像を使って学習してしまう。
        tmp=rem.copy()
        for y in range(H):
            for x in range(W):
                if rem[y,x]==0:
                    continue
                elif rem[y,x]==1:
                    print("check1={}, check2={}, check3={} ,check4={} ,check5={}".format(check1(x,y,tmp),check2(x,y,tmp),check3(x,y,tmp),check4(x,y,rem),check5(x,y,tmp)))
                    if check1(x,y,tmp)==1 and check2(x,y,tmp)==1 and check3(x,y,tmp)==1 and check4(x,y,rem)==1 and check5(x,y,tmp)==1:
                        rem[y,x]=-1

                num=np.where(rem==-1)
                if len(num[0])>0:
                    flag=1
                    rem[num]=0

        if flag==0:
            break


    out=np.zeros((H,W,C),dtype=np.float32)
    out[rem==1]=[255,255,255]
    out=out.astype(np.uint8)
    return out

img=cv2.imread("./input_image/gazo.png").astype(np.float32)
out=scanning(img)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output64.png",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
