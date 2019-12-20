import cv2
import numpy as np

def check1(x,y,out):
    if out[y,x]==0:
        return True
    return False

def check2(list):
    count=0
    for i in range(len(list)):
        if i==len(list)-1:
            if list[0]==1 and list[i]==0:
                count+=1
        else:
            if list[i+1]==1 and list[i]==0:
                count+=1
    if count==1:
        return True
    return False

def check3(list):
    list2=[i for i in list if i==1]
    if 2<=len(list2) and len(list2)<=6:
        return True
    return False


def check4_1(list):
    #list=[x2,x3,x4,x5,x6,x7,x8,x9]
    if list[0]==1 or list[2]==1 or list[4]==1:
        return True
    return False


def check5_1(list):
    #list=[x2,x3,x4,x5,x6,x7,x8,x9]
    if list[2]==1 or list[4]==1 or list[6]==1:
        return True
    return False


def check4_2(list):
    #list=[x2,x3,x4,x5,x6,x7,x8,x9]
    if list[0]==1 or list[2]==1 or list[6]==1:
        return True
    return False

def check5_2(list):
    #list=[x2,x3,x4,x5,x6,x7,x8,x9]
    if list[0]==1 or list[4]==1 or list[6]==1:
        return True
    return False


def zhang(img):
    H,W,C=img.shape
    #最後に反転させないといけない
    rem=np.ones((H,W),dtype=np.int) #white
    rem[img[:,:,0]>0]=0 #black

    while True:
        flag=0
        out=rem.copy()
        for y in range(H):
            for x in range(W):

                x9=out[max(0,y-1),max(0,x-1)]
                x2=out[max(0,y-1),x]
                x3=out[max(0,y-1),min(x+1,W-1)]
                x8=out[y,max(0,x-1)]
                x1=out[y,x]
                x4=out[y,min(W-1,x+1)]
                x7=out[min(H-1,y+1),max(0,x-1)]
                x6=out[min(y+1,H-1),x]
                x5=out[min(H-1,y+1),min(W-1,x+1)]
                list=[x2,x3,x4,x5,x6,x7,x8,x9]
                #Step1
                if check1(x,y,out) and check2(list) and check3(list) and check4_1(list) and check5_1(list):
                    rem[y,x]=1
                    flag=1

        out2=rem.copy()
        for y in range(H):
            for x in range(W):

                x9=out2[max(0,y-1),max(0,x-1)]
                x2=out2[max(0,y-1),x]
                x3=out2[max(0,y-1),min(x+1,W-1)]
                x8=out2[y,max(0,x-1)]
                x1=out2[y,x]
                x4=out2[y,min(W-1,x+1)]
                x7=out2[min(H-1,y+1),max(0,x-1)]
                x6=out2[min(y+1,H-1),x]
                x5=out2[min(H-1,y+1),min(W-1,x+1)]
                list=[x2,x3,x4,x5,x6,x7,x8,x9]
                #Step2
                if check1(x,y,out2) and check2(list) and check3(list) and check4_2(list) and check5_2(list):
                    rem[y,x]=1
                    flag=2

        print("flag={}".format(flag))
        if flag==0:
            break

    rem=np.abs(1-rem)
    rem*=255
    rem=rem.astype(np.uint8)

    return rem

img=cv2.imread("./input_image/gazo.png").astype(np.float32)
out=zhang(img)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output65.png",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
