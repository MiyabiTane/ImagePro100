import cv2
import numpy as np

def RGB_Ycbcr(img):
    H,W,C=img.shape
    out=np.zeros((H,W,C),dtype=np.float32)

    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y = 0.299 * R + 0.5870 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    out[:,:,0]=Y
    out[:,:,1]=Cb
    out[:,:,2]=Cr

    return out

def Ycbcr_RGB(img):
    H,W,C=img.shape
    out=np.zeros((H,W,C),dtype=np.float32)

    Y=img[:,:,0]
    Cb=img[:,:,1]
    Cr=img[:,:,2]

    R = Y + (Cr - 128) * 1.402
    G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    B = Y + (Cb - 128) * 1.7718

    out[:,:,0]=B
    out[:,:,1]=G
    out[:,:,2]=R

    out=np.clip(out,0,255)
    out=out.astype(np.uint8)
    return out

original=cv2.imread("./input_image/imori.jpg")
cv2.imshow("original",original)
img=cv2.imread("./input_image/imori.jpg").astype(np.float32)
YCbCr=RGB_Ycbcr(img)
YCbCr[:,:,0]*=0.7
out=Ycbcr_RGB(YCbCr)
cv2.imwrite("./output_image/output39.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
