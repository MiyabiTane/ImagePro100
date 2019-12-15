import cv2
import numpy as np

def SAD_matching(img,part):
    H,W,C=img.shape
    h,w,Ct=part.shape
    S_min=255*H*W*C

    for y in range(H-h):
        for x in range(W-w):
            S=np.sum(np.abs(img[y:y+h,x:x+w]-part))
            if S<S_min:
                S_min=S
                x_left=x
                x_right=x+w
                y_up=y
                y_bottom=y+h
    return x_left,x_right,y_up,y_bottom

img_ori=cv2.imread("./input_image/imori.jpg")
img=img_ori.astype(np.float32)
part=cv2.imread("./input_image/imori_part.jpg").astype(np.float32)
x_left,x_right,y_up,y_bottom=SAD_matching(img,part)
cv2.rectangle(img_ori,(x_left,y_up),(x_right,y_bottom),(0,0,255))
cv2.imshow("result",img_ori)
cv2.imwrite("./output_image/output55う.jpg",img_ori)
cv2.waitKey(0)
cv2.destroyAllWindows()
