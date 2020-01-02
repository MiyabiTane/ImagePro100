import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def gau_py(gray):
    map=[]
    for i in range(6):
        out=bl_interpolate(gray,ax=1/2**i,ay=1/2**i)
        #大きさ128にする
        out=bl_interpolate(out,ax=2**i,ay=2**i)
        map.append(out.astype(np.float32))
    return map

def kencho(map):
    list=[[0,1],[0,3],[0,5],[1,4],[2,3],[3,5]]
    H,W=map[0].shape
    out=np.zeros((H,W),dtype=np.float32)

    for i in range(len(list)):
        diff=np.abs(map[list[i][0]]-map[list[i][1]])
        out+=diff
    out=out*255/out.max()
    out=out.astype(np.uint8)
    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
gray=BGR2GRAY(img)
#画像をいじる時はfloat型に直す。
gray=gray.astype(np.float32)
map=gau_py(gray)
out=kencho(map)
cv2.imwrite("./output_image/output76.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
