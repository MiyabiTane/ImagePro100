import cv2

img=cv2.imread("imori.jpg")

B=img[:,:,0].copy()
G=img[:,:,1].copy()
R=img[:,:,2].copy()

Y = 0.2126*R + 0.7152*G + 0.0722*B

img[:,:,0]=Y
img[:,:,1]=Y
img[:,:,2]=Y

cv2.imwrite("./output_image/output2.jpg",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
