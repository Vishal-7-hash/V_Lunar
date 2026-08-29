
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(path, gray=True):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)

    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
    print("Min/Max:", img.min(), img.max())

    return img

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img



def visulize(source):
    plt.subplot(1,2,1)
    plt.title("Source")
    plt.imshow(source, cmap='gray')

    # plt.subplot(1,2,2)
    # plt.title("Reference")
    # plt.imshow(reference, cmap='gray')

    plt.show()