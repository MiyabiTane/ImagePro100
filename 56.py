import cv2
import numpy as np

def NCC_matching(img,part):
    H,W,C=img.shape
    h,w,Ct=part.shape
    S_max=-1

    for y in range(H-h):
        for x in range(W-w):
            numer=np.sum(img[y:y+h,x:x+w]*part)
            # Sqrt(Sum_{x=0:w, y=0:h} I(i+x, j+y)^2) * Sqrt(Sum_{x=0:w, y=0:h} T(x, y)^2)
            denom=np.sqrt(np.sum(img[y:y+h,x:x+w]**2)*np.sqrt(np.sum(part**2)))
            S=numer/denom
            if S>S_max:
                S_max=S
                x_left=x
                x_right=x+w
                y_up=y
                y_bottom=y+h
    return x_left,x_right,y_up,y_bottom

img_ori=cv2.imread("./input_image/imori.jpg")
img=img_ori.astype(np.float32)
part=cv2.imread("./input_image/imori_part.jpg").astype(np.float32)
x_left,x_right,y_up,y_bottom=NCC_matching(img,part)
cv2.rectangle(img_ori,(x_left,y_up),(x_right,y_bottom),(0,0,255))
cv2.imshow("result",img_ori)
cv2.imwrite("./output_image/output56.jpg",img_ori)
cv2.waitKey(0)
cv2.destroyAllWindows()
