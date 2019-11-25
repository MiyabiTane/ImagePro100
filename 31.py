import cv2
import numpy as np

def sharing(img,dx,dy,tx,ty):
    H,W,C=img.shape
    #0パッド
    img_e=np.zeros((H+2,W+2,C),dtype=np.float)
    img_e[1:H+1,1:W+1]=img

    aH=int(H+dy)
    aW=int(W+dx)
    out=np.zeros((aH+2,aW+2,C),dtype=np.float)
    #新しい画像上の位置
    y_d=np.arange(aH).repeat(aW).reshape(aH,-1)
    x_d=np.tile(np.arange(aW),(aH,1))
    #元画像上の変換後の位置
    a=dx/H
    b=dy/W
    x=((x_d-a*y_d)/(1-a*b)-tx*x_d).astype(np.int)
    y=((-b*x_d+y_d)/(1-a*b)-ty*y_d).astype(np.int)
    #out of range防止
    x=np.minimum(np.maximum(x,0),W+1).astype(np.int)
    y=np.minimum(np.maximum(y,0),H+1).astype(np.int)

    out[y_d,x_d]=img_e[y,x]
    out=out[1:aH,1:aW,:]
    out=np.clip(out,0,255)
    out=out.astype(np.uint8)

    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float)
img_1=sharing(img,dx=30,dy=0,tx=0,ty=0)
img_2=sharing(img,dx=0,dy=30,tx=0,ty=0)
img_3=sharing(img,dx=30,dy=30,tx=0,ty=0)
cv2.imwrite("./output_image/output31_1.jpg",img_1)
cv2.imwrite("./output_image/output31_2.jpg",img_2)
cv2.imwrite("./output_image/output31_3.jpg",img_3)
cv2.imshow("result1",img_1)
cv2.imshow("result2",img_2)
cv2.imshow("result3",img_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
