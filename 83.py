import cv2
import numpy as np
import matplotlib.pyplot as plt

def BGR2GRAY(img):
    out=img.copy()
    B=img[:,:,0].copy()
    G=img[:,:,1].copy()
    R=img[:,:,2].copy()

    Y=0.2126*R+0.7152*G+0.0722*B
    out=Y
    out=out.astype(np.uint8)
    return out

def sobel_fil(gray):
		# get shape
		H, W = gray.shape

		# sobel kernel
		sobely = np.array(((1, 2, 1),
						(0, 0, 0),
						(-1, -2, -1)), dtype=np.float32)

		sobelx = np.array(((1, 0, -1),
						(2, 0, -2),
						(1, 0, -1)), dtype=np.float32)

		# padding
		tmp = np.pad(gray, (1, 1), 'edge')

		# prepare
		Ix = np.zeros_like(gray, dtype=np.float32)
		Iy = np.zeros_like(gray, dtype=np.float32)

		# get differential
		for y in range(H):
			for x in range(W):
				Ix[y, x] = np.mean(tmp[y : y  + 3, x : x + 3] * sobelx)
				Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)
		return Ix, Iy

def gaussian_filter(I, K_size=3, sigma=3):
		# get shape
		H, W = I.shape

		## gaussian
		I_t = np.pad(I, (K_size // 2, K_size // 2), 'edge')

		# gaussian kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(K_size):
			for y in range(K_size):
				_x = x - K_size // 2
				_y = y - K_size // 2
				K[y, x] = np.exp( -(_x ** 2 + _y ** 2) / (2 * (sigma ** 2)))
		K /= (sigma * np.sqrt(2 * np.pi))
		K /= K.sum()

		# filtering
		for y in range(H):
			for x in range(W):
				I[y,x] = np.sum(I_t[y : y + K_size, x : x + K_size] * K)

		return I

def Harris_step1(gray,Iy,Ix):
    Ix2=gaussian_filter(Ix**2,K_size=3,sigma=3)
    Iy2=gaussian_filter(Iy**2,K_size=3,sigma=3)
    Ixy=gaussian_filter(Ix*Iy,K_size=3,sigma=3)

    return Ix2,Iy2,Ixy

def Harris_step2(gray,Ix2,Iy2,Ixy,k=0.04,th=0.1):
    H,W=gray.shape
    det_H=np.zeros((H,W),dtype=np.float32)
    trace_H=np.zeros((H,W),dtype=np.float32)
    R=np.zeros((H,W),dtype=np.float32)

    #可視化のため
    out=np.array((gray,gray,gray))
    out=np.transpose(out,(1,2,0))

    for y in range(H):
        for x in range(W):
            det_H[y,x]=Ix2[y,x]*Iy2[y,x]-(Ixy[y,x])**2
            trace_H[y,x]=Ix2[y,x]+Iy2[y,x]
            R[y,x]=det_H[y,x]-k*(trace_H[y,x]**2)
    #コーナーを赤で表示
    for y in range(H):
        for x in range(W):
            if R[y,x]>=np.max(R)*th:
                out[y,x]=[0,0,255]

    out=np.clip(out,0,255).astype(np.uint8)
    return out


img=cv2.imread("./input_image/thorino.jpg").astype(np.float32)
gray=BGR2GRAY(img)
Ix,Iy=sobel_fil(gray)
Ix2,Iy2,Ixy=Harris_step1(gray,Iy,Ix)
out=Harris_step2(gray,Ix2,Iy2,Ixy)
cv2.imwrite("./output_image/output83.jpg",out)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
