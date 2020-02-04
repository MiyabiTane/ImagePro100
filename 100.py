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

def IoU(c,d):
    a=np.array(c,dtype=np.float32)
    b=np.array(d,dtype=np.float32)
    area_a=(a[3]-a[1])*(a[2]-a[0])
    area_b=(b[3]-b[1])*(b[2]-b[0])
    Rol_wid=min(a[2],b[2])-max(a[0],b[0])
    Rol_hei=min(a[3],b[3])-max(a[1],b[1])
    if Rol_wid<0 or Rol_hei<0:
        return 0
    area_Rol=Rol_wid*Rol_hei
    IoU=np.abs(area_Rol)/np.abs(area_a+area_b-area_Rol)
    #print(IoU)
    return IoU

#データ作成
def cropping(img,num=200):
    H,W,C=img.shape
    np.random.seed(0)
    data=[]
    label=[]
    for i in range(num):
        x1=np.random.randint(W-60)
        y1=np.random.randint(H-60)
        x2=x1+60
        y2=y1+60

        GT=np.array((47,41,129,103),dtype=np.float32)
        list=np.array((x1,y1,x2,y2),dtype=np.float32)
        iou=IoU(GT,list)
        #データ
        hog_img=img[y1:y2,x1:x2]
        hog_img=cv2.resize(hog_img,(32,32))
        hist=GetHog(hog_img)
        h,w,c=hist.shape
        hist=hist.reshape([h*w*c])
        data.append(hist)
        #教師ラベル
        if iou>=0.5:
            label.append([1])
        else:
            label.append([0])
    data=np.array(data)
    label=np.array(label)
    print(data.shape)
    print(label.shape)
    return data,label

#データを用意
img=cv2.imread("./input_image/imori_1.jpg").astype(np.float32)
data,label=cropping(img)

#ニューラルネット
np.random.seed(0)
class NN:
    def __init__(self, ind=2, w=64, outd=1, lr=0.1):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.w2=np.random.normal(0,1,[w,w])
        self.b2=np.random.normal(0,1,[w])
        self.wout = np.random.normal(0, 1, [w, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3=sigmoid(np.dot(self.z2,self.w2)+self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_En = En #np.array([En for _ in range(t.shape[0])])
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout#np.expand_dims(grad_wout, axis=-1)
        self.bout -= self.lr * grad_bout

        # backpropagation middle layer
        grad_u2 = np.dot(grad_En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2

        # backpropagation inter layer
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

#データ
train_x = np.array(data,dtype=np.float32)
train_t = np.array(label,dtype=np.float32)

nn = NN(ind=train_x.shape[1],lr=0.01)
epoch=10000

#train
for epo in range(epoch):
    nn.forward(train_x)
    nn.train(train_x, train_t)

def FindObjects_Step1(img):
    H,W,C=img.shape
    get_hog=[]
    get_coordinates=[]

    for y in range(0,H,4):
        for x in range(0,W,4):
            #矩形42*42
            rects1=img[max(0,y-21):min(H+1,y+22),max(0,x-21):min(W+1,x+22)]
            rects1=cv2.resize(rects1,(32,32))
            hog1=GetHog(rects1)
            h,w,c=hog1.shape
            hog1=hog1.reshape([h*w*c])
            get_hog.append(hog1)
            get_coordinates.append([max(0,x-21),max(0,y-21),min(W+1,x+22),min(H+1,y+22),0.0])
            #矩形56*56
            rects2=img[max(0,y-28):min(H+1,y+29),max(0,x-28):min(W+1,x+29)]
            rects2=cv2.resize(rects2,(32,32))
            hog2=GetHog(rects2)
            h,w,c=hog2.shape
            hog2=hog2.reshape([h*w*c])
            get_hog.append(hog2)
            get_coordinates.append([max(0,x-28),max(0,y-28),min(W+1,x+29),min(H+1,y+29),0.0])
            #矩形70*70
            rects3=img[max(0,y-35):min(H+1,y+36),max(0,x-35):min(W+1,x+36)]
            rects3=cv2.resize(rects3,(32,32))
            hog3=GetHog(rects3)
            h,w,c=hog3.shape
            hog3=hog3.reshape([h*w*c])
            get_hog.append(hog3)
            get_coordinates.append([max(0,x-35),max(0,y-35),min(W+1,x+36),min(H+1,y+36),0.0])
    return np.array(get_coordinates),get_hog


def HighScoreOnly(imori_face,t=0.25):
    answer=[]
    #predの降順にソート
    sort_num=np.argsort(imori_face[:,-1])[::-1]
    new_face=imori_face[sort_num]
    print("new_face")
    print(new_face)
    B=new_face.tolist()
    #Iou
    while B:
        print(B)
        b0=[B[0][0],B[0][1],B[0][2],B[0][3]]
        b_an=[B[0][0],B[0][1],B[0][2],B[0][3],B[0][4]]
        if len(B)==1:
            B.pop(0)
            answer.append(b_an)
        else:
            for j in range(1,len(B)):
                b1=[B[j][0],B[j][1],B[j][2],B[j][3]]
                #print("b1")
                #print(b1)
                iou=IoU(b0,b1)
                if iou>=t:
                    B[j][0]=-1
            B.pop(0)
            B=[b for b in B if b[0]!=-1]
            answer.append(b_an)
    print("answer")
    print(answer)
    return np.array(answer,dtype=np.float32)


img2=cv2.imread("./input_image/imori_many.jpg").astype(np.float32)
get_coordinates,get_hog=FindObjects_Step1(img2)
#print(get_coordinates)
#print(get_coordinates.shape)
#test
imori_face=[]
for i in range(len(get_hog)):
    x=get_hog[i]
    pred=nn.forward(x)
    if pred>=0.7:
        print(pred)
        get_coordinates[i,-1]=pred
        imori_face.append(get_coordinates[i].tolist())
print(imori_face)
imori_face=np.array(imori_face)

answer=HighScoreOnly(imori_face)

def lookup(answer,gt,t=0.5):
    #table[i]=[gt1,gt2] t>=0.5なら1,t<0.5なら0
    table=[]
    for i in range(len(answer)):
        flag1=0
        iou=IoU(gt[0],answer[i,:-1])
        if iou>=t:
            flag1=1
        flag2=0
        iou=IoU(gt[1],answer[i,:-1])
        if iou>=t:
            flag2=1
        table.append([flag1,flag2])
    return np.array(table)


def recall(answer,table,length=2):
    G=length
    G_d=0
    for i in range(length):
        if 1 in table[:,i]:
            G_d+=1
    print("Recall >> {} ({}/{})".format(G_d/G,G_d,G))
    return G_d/G


def precision(answer,table):
    D=len(answer)
    D_d=0
    for i in range(D):
        if 1 in table[i,:]:
            D_d+=1
    print("Precision >> {} ({}/{})".format(D_d/D,D_d,D))
    return D_d/D


def f_score(rec,pre):
    f=2*rec*pre/(rec+pre)
    print("F-score >> {}".format(f))
    return f


def mAP(answer,table):
    map=0
    count=0
    for i in range(len(answer)):
        if 1 in table[i,:]: #detect=1
            map+=precision(answer[:i+1,:],table)
            count+=1
    print("mAP >> {}".format(map))
    return map


gt=np.array(((27,48,95,110),(101,75,171,138)),dtype=np.float32)
table=lookup(answer,gt)
print("table")
print(table)
rec=recall(answer,table)
pre=precision(answer,table)
f_score=f_score(rec,pre)
map=mAP(answer,table)
#矩形を描画
for i in range(len(gt)):
    cv2.rectangle(img2,(gt[i,0],gt[i,1]),(gt[i,2],gt[i,3]),(0,255,0))
for i in range(len(answer)):
    if 1 in table[i,:]:
        cv2.rectangle(img2,(answer[i,0],answer[i,1]),(answer[i,2],answer[i,3]),(0,0,255))
    else:
        cv2.rectangle(img2,(answer[i,0],answer[i,1]),(answer[i,2],answer[i,3]),(255,0,0))
    cv2.putText(img2, "{:.2f}".format(answer[i,-1]), (int(answer[i,0]), int(answer[i,1]+9)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)
img2=img2.astype(np.uint8)
cv2.imwrite("./output_image/output100.jpg",img2)
cv2.imshow("result",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
