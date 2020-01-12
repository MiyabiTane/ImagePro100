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

    plt.subplot(1,3,1)
    plt.imshow(Ix2,cmap='gray')
    plt.title("Ix^2")

    plt.subplot(1,3,2)
    plt.imshow(Iy2,cmap='gray')
    plt.title("Iy^2")

    plt.subplot(1,3,3)
    plt.imshow(Ixy,cmap='gray')
    plt.title("Ixy")

    plt.tight_layout()
    #plt.show()
    plt.savefig("./output_image/output82.png")

img=cv2.imread("./input_image/thorino.jpg").astype(np.float32)
gray=BGR2GRAY(img)
Ix,Iy=sobel_fil(gray)
Harris_step1(gray,Iy,Ix)
