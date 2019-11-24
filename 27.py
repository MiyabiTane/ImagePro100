import cv2
import numpy as np

def h(d,a=1.5):
    d=abs(d)
    if d<=1:
        return (a+2)*(d**3)-(a+3)*(d**2)+1
    elif 1<d and d<=2:
        return a*(d**3)-5*a*(d**2)+8*a*d-4*a
    else:
        return 0

def bi_cubic(img,a=1.5):
    H,W,C=img.shape
    aH=int(H*a);aW=int(W*a)
    out=np.zeros((aH,aW,C),dtype=np.float)

    for i in range(aH):
        for j in range(aW):
            x_a=i/a; y_a=j/a
            x=np.floor(x_a).astype(np.int)
            y=np.floor(y_a).astype(np.int)

            if x>W-2:
                x=W-2
            if y>H-2:
                y=H-2

            h_x=[h(x_a-(x-1)),h(x_a-x),h(x_a-(x+1)),h(x_a-(x+2))]
            h_y=[h(y_a-(y-1)),h(y_a-y),h(y_a-(y+1)),h(y_a-(y+2))]

            denom=0
            for k in range(4):
                for l in range(4):
                    out[j,i]+=img[y+k-2,x+l-2]*h_x[l]*h_y[k]
                    denom+=h_x[l]*h_y[k]
            out[j,i]/=denom
    out=np.clip(out,0,255)
    out=out.astype(np.uint8)
    return out

img=cv2.imread("./input_image/imori.jpg").astype(np.float)
img_n=bi_cubic(img)
cv2.imwrite("./output_image/output27.jpg",img_n)
cv2.imshow("result1",img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
