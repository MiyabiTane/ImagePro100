import cv2
import numpy as np

def link4(img):
    H,W,C=img.shape
    rem=np.zeros((H,W),dtype=np.int) #black
    rem[img[:,:,0]>0]=1 #not black
    colors=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[155,0,255]]

    for y in range(H):
        for x in range(W):
            if rem[y,x]==0:
                continue
            x4=rem[max(y-1,0),max(x-1,0)]
            x3=rem[max(y-1,0),x]
            x2=rem[max(0,y-1),min(x+1,W-1)]
            x5=rem[y,max(x-1,0)]
            x1=rem[y,min(x+1,W-1)]
            x6=rem[min(y+1,H-1),max(0,x-1)]
            x7=rem[min(y+1,H-1),x]
            x8=rem[min(y+1,H-1),min(x+1,W-1)]

            S=int((x1-x1*x2*x3)+(x3-x3*x4*x5)+(x5-x5*x6*x7)+(x7-x7*x8*x1))

            img[y,x]=colors[S]
    img=img.astype(np.uint8)
    return img

img=cv2.imread("./input_image/renketsu.png").astype(np.float32)
out=link4(img)
cv2.imshow("result",out)
cv2.imwrite("./output_image/output61.png",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
