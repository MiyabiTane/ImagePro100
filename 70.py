import cv2
import numpy as np

def BGR2HSV(img):
    #範囲を0~1にする
    img=img.copy()/255
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
    #img=(img*255).astype(np.uint8)
    return HSV


def color_track(hsv):
    H,W,C=hsv.shape
    out=np.zeros((H,W),dtype=np.float32)
    #青色トラッキング
    print(np.where((180<=hsv[:,:,0]) & (hsv[:,:,0]<=260)))
    out[np.where((180<=hsv[:,:,0]) & (hsv[:,:,0]<=260))]=255
    out=out.astype(np.uint8)

    return out


img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
hsv=BGR2HSV(img)
out=color_track(hsv)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output69.png",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
