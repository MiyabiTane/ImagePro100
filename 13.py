import cv2
import numpy as np

def gray_scale(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def max_min_filter(img):
    #W,H,C=img.shape
    if len(img.shape)==3:
        H,W,C=img.shape
    #要素が足りない場合は作り出す。
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #0パディング
    out=np.zeros((W+2,H+2,C),dtype=np.float)
    out[1:W+1,1:H+1]=img.copy()
    #フィルタリング
    out2=out.copy()
    for w in range(1,W+1):
        for h in range(1,H+1):
            for c in range(C):
                out2[w,h,c]=np.max(out[w-1:w+2,h-1:h+2,c])-np.min(out[w-1:w+2,h-1:h+2,c])
    out2=out2[1:W+1,1:H+1].astype(np.uint8)
    return out2

img=cv2.imread("imori.jpg").astype(np.float)
gray_img=gray_scale(img)
out=max_min_filter(gray_img)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output13.jpg",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
