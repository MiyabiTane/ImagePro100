import cv2

img=cv2.imread("imori.jpg")

B=img[:,:,0].copy()
G=img[:,:,1].copy()
R=img[:,:,2].copy()

Y = 0.2126*R + 0.7152*G + 0.0722*B

"""if img[Y]<128:
    img[Y]=0
else:
    img[Y]=255"""
img[Y<128]=0
img[Y>=128]=255

cv2.imwrite("output3.jpg",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
