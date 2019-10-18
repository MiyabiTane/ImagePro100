#課題３では閾値が128だった。今回はこの閾値を自分で決定する。
import cv2
import numpy as np

img=cv2.imread("imori.jpg")
#print("normal")
#print(img)
B=img[:,:,0].copy()
G=img[:,:,1].copy()
R=img[:,:,2].copy()
#print("B")
#print(img[:,:,0])

img=0.2126*R + 0.7152*G + 0.0722*B
#グレースケールにした画像
img=img.astype(np.uint8) #uint8->8ビット符号なし整数
#print(Y)

"""
print("Gray")
img[img<128]=0
img[img>=128]=1
print(img)
"""
sb2_max=0
t_max=0
for t in range(1,256):
    w0_list=img[np.where(img<t)]
    w1_list=img[np.where(img>=t)]
    w0=len(w0_list)/(len(w0_list)+len(w1_list))
    w1=len(w1_list)/(len(w0_list)+len(w1_list))
    #s0_2=np.var(w0_list)
    #s1_2=np.var(w1_list)
    #場合分けしないとエラー
    m0=np.mean(w0_list) if len(w0_list)>0 else 0
    m1=np.mean(w1_list) if len(w1_list)>0 else 0

    sb_2=w0*w1*((m0-m1)**2)
    if sb_2 > sb2_max:
        sb2_max=sb_2
        t_max=t
print(t_max)

img[img<t_max]=0
img[img>=t_max]=255

cv2.imwrite("./output_image/output4.jpg",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
