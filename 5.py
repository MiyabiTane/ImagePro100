import cv2
import numpy as np

img=cv2.imread("imori.jpg")

B=img[:,:,0].copy()
G=img[:,:,1].copy()
R=img[:,:,2].copy()

#RGB->HSV
Max=max(R.all(),G.all(),B.all())
Min=min(R.all(),G.all(),B.all())

if Min==Max:
    H=0
elif Min==B.all():
    H=60*(G-R)/(Max-Min)+60
elif Min==R.all():
    H=60*(B-G)/(Max-Min)+180
else:
    H=60*(R-B)/(Max-Min)+300
V=Max
S=Max-Min
#反転
H+=180
#HSV->RGB
C=S
H/=60
X=C(1-abs(H%2-1))

if 0<=H and H<1:
    (R,G,B)=(V-C)(1,1,1)+(C,X,0)
elif 1<=H and H<2:
    (R,G,B)=(V-C)(1,1,1)+(X,C,0)
elif 2<=H and H<3:
    (R,G,B)=(V-C)(1,1,1)+(0,C,X)
elif 3<=H and H<4:
    (R,G,B)=(V-C)(1,1,1)+(0,X,C)
elif 4<=H and H<5:
    (R,G,B)=(V-C)(1,1,1)+(X,0,C)
elif 5<=H and H<6:
    (R,G,B)=(V-C)(1,1,1)+(C,0,X)

img[:,:,0]=B
img[:,:,1]=G
img[:,:,2]=R

cv2.imwrite("./output_image/output5.jpg",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
