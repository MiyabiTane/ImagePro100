import cv2
import numpy as np

def BGR2HSV(img):
    #範囲を0~1にする
    img=img.copy()/255
    #imgと同じ型の０配列生成
    HSV=np.zeros_like(img,dtype=np.float32)
    #RGBのうちどれが一番大きいor小さいか。代入されるのはRGBどれかの配列
    Max=np.max(img,axis=2).copy()
    Min=np.min(img,axis=2).copy()
    #RGBのうち、最小のもののインデックスを返す。
    #https://deepage.net/features/numpy-argmin.html
    min_arg=np.argmin(img,axis=2)

    HSV[:,:,0][np.where(Max==Min)]=0
    if1=np.where(min_arg==0)
    HSV[:,:,0][if1]=60*(img[:,:,1][if1]-img[:,:,2][if1])/(Max[if1]-Min[if1])+60
    if2=np.where(min_arg==2)
    HSV[:,:,0][if2]=60*(img[:,:,0][if2]-img[:,:,1][if2])/(Max[if2]-Min[if2])+180
    if3=np.where(min_arg==1)
    HSV[:,:,0][if3]=60*(img[:,:,2][if3]-img[:,:,0][if3])/(Max[if3]-Min[if3])+300

    HSV[:,:,2]=Max.copy()
    HSV[:,:,1]=Max.copy()-Min.copy()
    #img=(img*255).astype(np.uint8)
    return HSV


def masking(img,hsv):
    H,W,C=hsv.shape
    mask=np.zeros((H,W),dtype=np.float32)
    mask_n=np.zeros((H,W,C),dtype=np.float32)
    #青色->1、その他->0
    #print(np.where((180<=hsv[:,:,0]) & (hsv[:,:,0]<=260)))
    mask[np.where((180<=hsv[:,:,0]) & (hsv[:,:,0]<=260))]=1
    cv2.imshow("mask",mask)

    mask=closing(mask,N=5)
    mask=opening(mask,N=5)
    cv2.imshow("mask2",mask)
    cv2.imwrite("./output_image/output72_1.png",mask)

    mask_n[np.where(mask==1)]=1
    #print(mask_n)

    out=img*(1-mask_n)

    out=out.astype(np.uint8)
    return out

#-----注意：maskに対して行うので255->1に置き換える必要がある。-----
# Dilation
def Dilate(img, time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H + 1):
			for x in range(1, W + 1):
				if np.sum(MF * tmp[y - 1 : y + 2, x - 1 : x + 2]) >= 1:
					out[y - 1, x - 1] = 1

	return out



def dilate(img,time=2):
    H,W=img.shape
    out=img.copy()

    for t in range(time):
        #端の値をコピーしてパディング
        out_2=np.pad(out,(1,1),'edge')
        for y in range(1,H+1):
            for x in range(1,W+1):
                if out_2[y,x]==0:
                    if out_2[y-1,x]==1 or out_2[y,x-1]==1 or out_2[y,x+1]==1 or out_2[y+1,x]==1:
                        out[y-1,x-1]=1
    #img=img.astype(np.uint8)
    return out

def erode(img,time=2):
    H,W=img.shape
    out=img.copy()

    for t in range(time):
        pad_img=np.pad(out,(1,1),'edge')
        for y in range(1,H+1):
            for x in range(1,W+1):
                if pad_img[y,x]==1:
                    if pad_img[y-1,x]==0 or pad_img[y,x-1]==0 or pad_img[y,x+1]==0 or pad_img[y+1,x]==0:
                        out[y-1,x-1]=0
    return out


def opening(img,N=5):
    img=erode(img,time=N)
    img=dilate(img,time=N)
    return img

def closing(img,N=5):
    img=dilate(img,time=N)
    img=erode(img,time=N)
    return img


img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
hsv=BGR2HSV(img)
out=masking(img,hsv)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output72_2.jpg",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
