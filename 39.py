import cv2
import numpy as np

def Ycbcr(img,a=0.7):
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y = 0.299 * R + 0.5870 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    Y*=a

    R = Y + (Cr - 128) * 1.402
    G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    B = Y + (Cb - 128) * 1.7718

    img[:,:,0]=B
    img[:,:,1]=G
    img[:,:,2]=R

    img=np.clip(img,0,255)
    img=img.astype(np.uint8)
    return img

img=cv2.imread("./input_image/imori.jpg")
cv2.imshow("original",img)
img=img.astype(np.uint8)
out=Ycbcr(img)
cv2.imwrite("./output_image/output39.jpg",out)
cv2.imshow("original",img)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
