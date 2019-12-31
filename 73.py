import cv2
import numpy as np

def BGR2GRAY(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out


def bi_linear(img,a=1.5):
    H,W=img.shape
    aH=int(H*a);aW=int(W*a)
    out=np.zeros((aH,aW),dtype=np.float)

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

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
gray=BGR2GRAY(img)
small=bi_linear(gray,a=0.5)
out=bi_linear(small,a=2.0)
cv2.imwrite("./output_image/output73.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
