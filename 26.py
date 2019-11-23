import cv2
import numpy as np

def bi_linear(img,a=1.5):
    H,W,C=img.shape
    aH=int(H*a);aW=int(W*a)
    out=np.zeros((aH,aW,C),dtype=np.float)
    """
    x_d=np.arange(aH).repeat(aW).reshape(aW,-1)
    y_d=np.tile(np.arange(aW),(aH,1))
    #元画像上での位置
    x_a=np.floor(x_d/a)
    y_a=np.floor(y_d/a)
    """
    for i in range(aH):
        for j in range(aW):
            x_a=i/a;y_a=j/a
            #floorをとれば一番近くの自分より小さい価がとってこれる。
            x=np.floor(x_a).astype(np.int)
            y=np.floor(y_a).astype(np.int)
            #dx,dyを求める。イメージはここ参照 ▶︎ https://algorithm.joho.info/image-processing/bi-linear-interpolation/
            dx=x_a-x;dy=y_a-y
            if x>W-2:
                x=W-2
            if y>H-2:
                y=H-2
            out[j,i]=(1-dy)*(1-dx)*img[y,x]+dx*(1-dx)*img[y+1,x]+(1-dy)*dx*img[y,x+1]+dx*dy*img[y+1,x+1]
    out=np.clip(out,0,255)
    out=out.astype(np.uint8)

    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float)
img_n=bi_linear(img)
cv2.imwrite("./output_image/output26.jpg",img_n)
cv2.imshow("result1",img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
