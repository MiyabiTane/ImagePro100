import numpy as np
import cv2

def IoU(a,b):
    area_a=(a[3]-a[1])*(a[2]-a[0])
    area_b=(b[3]-b[1])*(b[2]-b[0])
    Rol_wid=min(a[2],b[2])-max(a[0],b[0])
    Rol_hei=min(a[3],b[3])-max(a[1],b[1])
    if Rol_wid<0 or Rol_hei<0:
        return 0
    area_Rol=Rol_wid*Rol_hei
    IoU=np.abs(area_Rol)/np.abs(area_a+area_b-area_Rol)
    #print(IoU)
    return IoU

def cropping(img,num=200):
    H,W,C=img.shape
    np.random.seed(0)
    rects=[]
    for i in range(num):
        x1=np.random.randint(W-60)
        y1=np.random.randint(H-60)
        x2=x1+60
        y2=y1+60

        GT=np.array((47,41,129,103),dtype=np.float32)
        list=np.array((x1,y1,x2,y2),dtype=np.float32)
        iou=IoU(GT,list)

        if iou>=0.5:
            rects.append([x1,y1,x2,y2,1])
        else:
            rects.append([x1,y1,x2,y2,0])
    rects=np.array(rects)
    print(rects)
    #描写
    for i in range(num):
        if rects[i,-1]==1:
            cv2.rectangle(img,(rects[i,0],rects[i,1]),(rects[i,2],rects[i,3]),(0,0,255))
        else:
            cv2.rectangle(img,(rects[i,0],rects[i,1]),(rects[i,2],rects[i,3]),(255,0,0))
    cv2.rectangle(img,(GT[0],GT[1]),(GT[2],GT[3]),(0,255,0))
    img=img.astype(np.uint8)
    return img

img=cv2.imread("./input_image/imori_1.jpg").astype(np.float32)
out=cropping(img)
cv2.imwrite("./output_image/output94.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
