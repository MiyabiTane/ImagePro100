import cv2
import numpy as np

def expand_i(img,a=1.5):
    H,W,C=img.shape
    #out=np.zeros((int(H*a),int(W*a),C),dtype=np.float)
    aH=int(H*a);aW=int(W*a)

    x=np.arange(aH).repeat(aW).reshape(aW,-1)
    y=np.tile(np.arange(aW),(aH,1))
    x=np.round(x/a).astype(np.int)
    y=np.round(y/a).astype(np.int)

    out=img[x,y]
    out=out.astype(np.uint8)

    return out

img=cv2.imread("imori.jpg").astype(np.float)
img_n=expand_i(img)
cv2.imwrite("./output_image/output25.jpg",img_n)
cv2.imshow("result1",img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""---メモ---
詳しくは25-test.pyを見ること
>>> aH=3;aW=3
>>> y = np.arange(aH).repeat(aW).reshape(aW, -1)
>>> y
array([[0, 0, 0],
       [1, 1, 1],
       [2, 2, 2]])
>>> x = np.tile(np.arange(aW), (aH, 1))
>>> x
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]])
"""
