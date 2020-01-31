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

def IoU(a,b):
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
# train
for epo in range(epoch):
    nn.forward(train_x)
    nn.train(train_x, train_t)

# test
count=0
for j in range(200):
    x = train_x[j]
    t = train_t[j]
    pred=nn.forward(x)
    if pred>=0.05:
        pred=1
    else:
        pred=0
    if pred==t:
        count+=1
    print(count)
accuracy=count/200
print("Accuracy >> {} ({}/200)".format(accuracy,count))
