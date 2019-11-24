import cv2
import numpy as np

def affine(img,a=1,b=0,c=0,d=1,tx=30,ty=-30):
    H,W,C=img.shape
    #周り１行0パッド
    img_e=np.zeros((H+2,W+2,C),dtype=np.float)
    img_e[1:H+1,1:W+1]=img

    aH=int(H*d)
    aW=int(W*a)
    out=np.zeros((aH+1,aW+1,C),dtype=np.float)
    #新しい画像の位置
    y_d=np.arange(aH).repeat(aW).reshape(aW,-1)
    x_d=np.tile(np.arange(aW),(aH,1))
    #元の画像上のAffine変換後の位置
    x=((d*x_d-b*y_d)/(a*d-b*c)).astype(np.int)-tx
    y=((-c*x_d+a*y_d)/(a*d-b*c)).astype(np.int)-ty

    #out of range を防ぐため
    x=np.minimum(np.maximum(x,0),W+1).astype(np.int)
    y=np.minimum(np.maximum(y,0),H+1).astype(np.int)

    out[y_d,x_d]=img_e[y,x]

    out=out[:aH,:aW,:]
    out=np.clip(out,0,255)
    out=out.astype(np.uint8)

    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float)
img_n=affine(img)
cv2.imwrite("./output_image/output28.jpg",img_n)
cv2.imshow("result1",img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
