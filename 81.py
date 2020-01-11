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

def sobel_fil(gray):
		# get shape
		H, W = gray.shape

		# sobel kernel
		sobely = np.array(((1, 2, 1),
						(0, 0, 0),
						(-1, -2, -1)), dtype=np.float32)

		sobelx = np.array(((1, 0, -1),
						(2, 0, -2),
						(1, 0, -1)), dtype=np.float32)

		# padding
		tmp = np.pad(gray, (1, 1), 'edge')

		# prepare
		Ix = np.zeros_like(gray, dtype=np.float32)
		Iy = np.zeros_like(gray, dtype=np.float32)

		# get differential
		for y in range(H):
			for x in range(W):
				Ix[y, x] = np.mean(tmp[y : y  + 3, x : x + 3] * sobelx)
				Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)
		return Ix, Iy

"""
def sobel_fil(img):
    if len(img.shape)==3:
        H,W,C=img.shape
    else:
        img=np.expand_dims(img,axis=-1)
        H,W,C=img.shape
    #0パディング
    out=np.zeros((H+2,W+2,C),dtype=np.float)
    out[1:H+1,1:W+1]=img.copy().astype(np.uint8)
    #縦方向
    K1=[[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]
    out_tate=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out_tate[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K1)
    out_tate=np.clip(out_tate,0,255)
    out_tate=out_tate[1:H+1,1:W+1].astype(np.uint8)
    #横方向
    K2=[[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]]
    out_yoko=out.copy()
    for h in range(1,H+1):
        for w in range(1,W+1):
            for c in range(C):
                out_yoko[h,w,c]=np.sum(out[h-1:h+2,w-1:w+2,c]*K2)
    out_yoko=np.clip(out_yoko,0,255)
    out_yoko=out_yoko[1:H+1,1:W+1].astype(np.uint8)

    return out_tate,out_yoko
"""

def Hessian(img,Iy,Ix):
    H,W=img.shape
    #out=np.zeros((H,W,3),dtype=np.float32)
    #可視化のため
    out=np.array((gray,gray,gray))
    out=np.transpose(out,(1,2,0))

    det_H=np.zeros((H,W),dtype=np.float32)
    #det(H)を求める
    for y in range(H):
        for x in range(W):
            det_H[y,x]=(Ix[y,x]**2)*(Iy[y,x]**2)-(Ix[y,x]*Ix[y,x])**2
    #極大点を探す
    for y in range(H):
        for x in range(W):
            if np.max(det_H[max(y-1,0):min(y+2,H+1),max(x-1,0):min(x+2,W+1)])==det_H[y,x] and det_H[y,x]>np.max(det_H)*0.1:
                out[y,x]=[0,0,255]
    out=out.astype(np.uint8)
    return out


img=cv2.imread("./input_image/thorino.jpg").astype(np.float32)
gray=BGR2GRAY(img)
Iy,Ix=sobel_fil(gray)
out=Hessian(gray,Iy,Ix)
cv2.imwrite("./output_image/output81.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""np.transposeメモ
>>> a=np.array([[1,2],[3,4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> out=np.array((a,a,a))
>>> out
array([[[1, 2],
        [3, 4]],

       [[1, 2],
        [3, 4]],

       [[1, 2],
        [3, 4]]])
>>> out=np.transpose(out,(1,2,0))
>>> out
array([[[1, 1, 1],
        [2, 2, 2]],

       [[3, 3, 3],
        [4, 4, 4]]])
"""
