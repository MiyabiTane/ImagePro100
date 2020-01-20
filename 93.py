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
    print(IoU)
    return IoU

a=np.array((50,50,150,150),dtype=np.float32)
b=np.array((60,60,170,160),dtype=np.float32)
IoU(a,b)
