import numpy as np
import cv2
import matplotlib.pyplot as plt

def gamma(img,c=1,g=2.2):
    #正規化を忘れずに
    out=img.copy()/255.
    out=(out/c)**(1/g)
#元(0~255)に戻す
    out*=255
    out=out.astype(np.uint8)
    return out

img=cv2.imread("imori_gamma.jpg").astype(np.float)
img_n=gamma(img)
cv2.imwrite("./output_image/output24.jpg",img_n)
cv2.imshow("result1",img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
