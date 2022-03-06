import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import canny, corner_harris, corner_peaks, corner_subpix
import glob
import cv2
import os
import numpy as np

def plot_canny(_image, path):

    image = rgb2gray(_image)
    fontdict = {
        'fontsize': 4
        }

    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(2,5))
    cc = canny(image, sigma=2.2)

    axes[0].imshow(cc, cmap=plt.cm.gray)
    axes[0].set_title('original', fontdict=fontdict)

    harris = np.zeros(cc.shape)
    for s in [1, 2, 3]:
        for k in [0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2]:
            harris = np.add(harris, corner_harris(cc, k=k, sigma=s))

    t_coords = corner_peaks(harris, min_distance=10, threshold_rel=0.2)
    coords_subpix = corner_subpix(image, t_coords, window_size=20)

    axes[1].imshow(harris, cmap=plt.cm.gray)
    axes[1].set_title('Výstup Harrisova \n detektoru', fontdict=fontdict)

    axes[2].imshow(_image, cmap=plt.cm.gray)
    axes[2].plot(coords_subpix[:, 1], coords_subpix[:, 0], '.r', markersize=1)
    axes[2].set_title('pozice rohu', fontdict=fontdict)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close('all')

gln = glob.glob("d:/marked/cropped_*.png")
print(len(gln))
for img_path in gln:
    new_path = "d:/harris/" + os.path.basename(img_path).replace("png","jpg")
    json_path = img_path.replace('.png', '.json')
    print(img_path)
    image = cv2.imread(img_path)
    plot_canny(image, new_path)