import cv2
img=cv2.imread("imori.jpg")

blue=img[:,:,0].copy()
green=img[:,:,1].copy()
red=img[:,:,2].copy()

img[:,:,0]=red
img[:,:,2]=blue

cv2.imwrite("./output_image/output1.jpg",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
