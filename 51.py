import cv2
import numpy as np

def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

def otsu_nichi(img):
    sb2_max=0
    t_max=0
    img=BGR2GRAY(img)
    for t in range(1,256):
        w0_list=img[np.where(img<t)]
        w1_list=img[np.where(img>=t)]
        w0=len(w0_list)/(len(w0_list)+len(w1_list))
        w1=len(w1_list)/(len(w0_list)+len(w1_list))
    #s0_2=np.var(w0_list)
    #s1_2=np.var(w1_list)
    #場合分けしないとエラー
        m0=np.mean(w0_list) if len(w0_list)>0 else 0
        m1=np.mean(w1_list) if len(w1_list)>0 else 0

        sb_2=w0*w1*((m0-m1)**2)
        if sb_2 > sb2_max:
            sb2_max=sb_2
            t_max=t

    img[img<t_max]=0
    img[img>=t_max]=255

    return img

def mor_expansion(img,time=2):
    H,W=img.shape
    out=img.copy()

    for t in range(time):
        #端の値をコピーしてパディング
        out_2=np.pad(out,(1,1),'edge')
        for y in range(1,H+1):
            for x in range(1,W+1):
                if out_2[y-1,x]==255 or out_2[y,x-1]==255 or out_2[y,x+1]==255 or out_2[y+1,x]==255:
                    out[y-1,x-1]=255

    #img=img.astype(np.uint8)
    return out


def mor_contraction(img,time=2):
    H,W=img.shape
    out=img.copy()

    for t in range(time):
        pad_img=np.pad(out,(1,1),'edge')
        for y in range(1,H+1):
            for x in range(1,W+1):
                if pad_img[y-1,x]==0 or pad_img[y,x-1]==0 or pad_img[y,x+1]==0 or pad_img[y+1,x]==0:
                    out[y-1,x-1]=0
    return out

def mor_slope(img):
    ex_img=mor_expansion(img,time=1)
    co_img=mor_contraction(img,time=1)
    out=np.abs(ex_img-co_img)
    return out

img = cv2.imread("./input_image/imori.jpg").astype(np.float32)
niti=otsu_nichi(img)
out=mor_slope(niti)

cv2.imshow("result2",out)
cv2.imwrite("./output_image/output51.jpg",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
