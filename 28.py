import cv2
import numpy as np

def affine_hor(img,tx=30,ty=-30):
    H,W,C=img.shape
    out=np.zeros((H+2,W+2,C),dtype=np.float)

    for i in range(H):
        for j in range(W):
            #元画像上の座標
            [x,y]=np.dot([[1,0],[0,1]],[i,j])-[tx,ty]
            #変換後の座標
            new=np.dot([[1,0,tx],[0,1,ty],[0,0,1]],[x,y,1])
            out[j,i]=img[new[1],new[0]]

    out=out[1:H+1,1:W+1,:]
    out=np.clip(out,0,255)
    out=out.astype(np.uint8)
    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float)
img_n=affine_hor(img)
cv2.imwrite("./output_image/output28.jpg",img_n)
cv2.imshow("result1",img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
