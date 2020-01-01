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

def bl_interpolate(img, ax=1., ay=1.):
	if len(img.shape) > 2:
		H, W, C = img.shape
	else:
		H, W = img.shape
		C = 1

	aH = int(ay * H)
	aW = int(ax * W)

	# get position of resized image
	y = np.arange(aH).repeat(aW).reshape(aW, -1)
	x = np.tile(np.arange(aW), (aH, 1))

	# get position of original position
	y = (y / ay)
	x = (x / ax)

	ix = np.floor(x).astype(np.int)
	iy = np.floor(y).astype(np.int)

	ix = np.minimum(ix, W-2)
	iy = np.minimum(iy, H-2)

	# get distance
	dx = x - ix
	dy = y - iy

	if C > 1:
		dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)
		dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)

	# interpolation
	out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out
"""
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
"""
def pyramid(gray,out):
    #out=out.astype(np.float32)
    pyr=np.abs(out-gray)
    #pyr=np.clip(pyr,0,255)
    pyr=pyr*255/pyr.max()
    pyr=pyr.astype(np.uint8)
    return pyr

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
gray=BGR2GRAY(img)
#画像をいじる時はfloat型に直す。
gray=gray.astype(np.float32)
small=bl_interpolate(gray, ax=0.5, ay=0.5)
out=bl_interpolate(small, ax=2.0, ay=2.0)
cv2.imshow("result1",out)
pyr=pyramid(gray,out)
cv2.imwrite("./output_image/output74.jpg",pyr)
cv2.imshow("result",pyr)
cv2.waitKey(0)
cv2.destroyAllWindows()
