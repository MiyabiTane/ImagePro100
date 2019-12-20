import cv2
import numpy as np

def gray_scale(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    #out=out.astype(np.uint8)
    return out

def hog_step1(img):
    H,W=img.shape
    mag=np.zeros((H,W),dtype=np.float32)
    ang=np.zeros((H,W),dtype=np.float32)
    for y in range(H):
        for x in range(W):
            gx=img[y,min(W-1,x+1)]-img[y,max(0,x-1)]
            #0除算を防ぐため
            if gx==0:
                gx=1e-6
            gy=img[min(H-1,y+1),x]-img[max(0,y-1),x]
            mag[y,x]=np.sqrt(gx**2+gy**2)
            ang[y,x]=np.arctan(gy/gx)
            if ang[y,x]<0:
                ang[y,x]+=np.pi

    th_ang=np.pi/9
    for i in range(9):
        ang[np.where((th_ang*i<=ang) & (ang<=th_ang*(i+1)))]=i

    return H,W,mag,ang

img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
gray=gray_scale(img)
H,W,mag,ang=hog_step1(gray)
mag=mag.astype(np.uint8)

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
     [127, 127, 0], [127, 0, 127], [0, 127, 127]]

out=np.zeros((H,W,3)).astype(np.uint8)
for i in range(9):
    out[ang==i]=colors[i]

cv2.imshow("result",mag)
cv2.imshow("result2",out)
cv2.imwrite("./output_image/output66_1.jpg",mag)
cv2.imwrite("./output_image/output66_2.jpg",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
