import cv2
import numpy as np

img_b=cv2.imread("imori.jpg")#.astype(np.float32)
#範囲を0~1にする
img=img_b.copy()/255
#imgと同じ型の０配列生成
HSV=np.zeros_like(img,dtype=np.float32)
#RGBのうちどれが一番大きいor小さいか。代入されるのはRGBどれかの配列
Max=np.max(img,axis=2).copy()
Min=np.min(img,axis=2).copy()
#RGBのうち、最小のもののインデックスを返す。
#https://deepage.net/features/numpy-argmin.html
min_arg=np.argmin(img,axis=2)

HSV[:,:,0][np.where(Max==Min)]=0
if1=np.where(min_arg==0)
HSV[:,:,0][if1]=60*(img[:,:,1][if1]-img[:,:,2][if1])/(Max[if1]-Min[if1])+60
if2=np.where(min_arg==2)
HSV[:,:,0][if2]=60*(img[:,:,0][if2]-img[:,:,1][if2])/(Max[if2]-Min[if2])+180
if3=np.where(min_arg==1)
HSV[:,:,0][if3]=60*(img[:,:,2][if3]-img[:,:,0][if3])/(Max[if3]-Min[if3])+300

HSV[:,:,2]=Max.copy()
HSV[:,:,1]=Max.copy()-Min.copy()

#変換
HSV[:,:,0]=(HSV[:,:,0]+180)%360

C=HSV[:,:,1]
H=HSV[:,:,0]
H_=H/60.
#X=np.dot(C,(1-abs(HSV[:,:,0]%2-1)))
X=C*(1-np.abs(H_%2-1))
#H=HSV[:,:,0]
V=HSV[:,:,2]
zero=np.zeros_like(H)
i_list=[[C,X,zero],[X,C,zero],[zero,C,X],[zero,X,C],[X,zero,C],[C,zero,X]]
#i_list= [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

for i in range(6):
    #&はビット演算子(論理和) andはダメ
    ifnum=np.where((i<=H_) & (H_<(i+1)))
    #R
    img[:,:,0][ifnum]=(V-C)[ifnum]+i_list[i][0][ifnum]
    #G
    img[:,:,1][ifnum]=(V-C)[ifnum]+i_list[i][1][ifnum]
    #R
    img[:,:,2][ifnum]=(V-C)[ifnum]+i_list[i][2][ifnum]
img[np.where(Max==Min)]=0
#1~255に戻す
img=np.clip(img,0,1) #全ての数字を0と1の間に収める。
#astypeしないと真っ白になる。
img=(img*255).astype(np.uint8)
#print(img)

"""
zero=np.zeros_like(H)
img[np.where(Max==Min)]=0
num=np.where(0<=H.all() and H.all()<1)
img[num]=np.dot((V[num]-C[num]),[1,1,1])+[C[num],X[num],zero]
num=np.where(1<=H.all() and H.all()<2)
img[num]=np.dot((V[num]-C[num]),[1,1,1])+[X[num],C[num],zero]
num=np.where(2<=H.all() and H.all()<3)
img[num]=np.dot((V[num]-C[num]),[1,1,1])+[zero,C[num],X[num]]
num=np.where(3<=H.all() and H.all()<4)
img[num]=np.dot((V[num]-C[num]),[1,1,1])+[zero,X[num],C[num]]
num=np.where(4<=H.all() and H.all()<5)
img[num]=np.dot((V[num]-C[num]),[1,1,1])+[X[num],zero,C[num]]
num=np.where(5<=H.all() and H.all()<6)
img[num]=np.dot((V[num]-C[num]),[1,1,1])+[C[num],zero,X[num]]
"""

cv2.imwrite("./output_image/output5.jpg",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
