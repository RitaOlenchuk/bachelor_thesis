from skimage import io
import random
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage import transform
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.feature import match_descriptors, corner_peaks, corner_harris, plot_matches, BRIEF
from skimage.measure import ransac
import matplotlib.pyplot as plt
import numpy as np
#https://www.researchgate.net/publication/264197576_scikit-image_Image_processing_in_Python
img1 = rgb2gray(io.imread('/usr/local/hdd/aorta_images/AR ZT13 8-1/AR_ZT13_8_1_rotated.jpg'))
img2 = rgb2gray(io.imread('/usr/local/hdd/aorta_images/HE/ZT13 8-1.tif'))

#img1 = rgb2gray(io.imread('/usr/local/hdd/rita/DL/sequence/ZT13_4-1.tif.small.tif'))
#img2 = rgb2gray(io.imread('/usr/local/hdd/rita/DL/sequence/ZT13_5-1.tif.small.tif'))

img1 = transform.rescale(img1, 0.1, multichannel=False)
img2 = transform.rescale(img2, 0.15, multichannel=False)

random.seed(0)

#ORB
'''
orb = ORB(n_keypoints=1000, fast_threshold=0.05)

orb.detect_and_extract(img1)
keypoints1 = orb.keypoints
desciptors1 = orb.descriptors

orb.detect_and_extract(img2)
keypoints2 = orb.keypoints
desciptors2 = orb.descriptors

matches12 = match_descriptors(desciptors1, desciptors2, cross_check=True)
'''
#BRIEF

keypoints1 = corner_peaks(corner_harris(img1), min_distance=5)
keypoints2 = corner_peaks(corner_harris(img2), min_distance=5)

extractor = BRIEF()

extractor.extract(img1, keypoints1)
keypoints1 = keypoints1[extractor.mask]
descriptors1 = extractor.descriptors

extractor.extract(img2, keypoints2)
keypoints2 = keypoints2[extractor.mask]
descriptors2 = extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

src = keypoints2 [ matches12[:, 1]][:, ::-1]
dst = keypoints1 [ matches12[:, 0]][:, ::-1]

model_robust, inliers = \
    ransac((src, dst), transform.SimilarityTransform, min_samples=4, residual_threshold=2)

r, c = img2.shape[:2]

corners = np.array([[0, 0],
    [0, r],
    [c, 0],
[c,r]])

warped_corners = model_robust(corners)
all_corners = np.vstack((warped_corners, corners))

corner_min = np.min(all_corners, axis=0)
corner_max = np.max(all_corners, axis=0)

output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1])

offset = transform.SimilarityTransform(translation=-corner_min)

img1_ = warp(img1, offset.inverse, output_shape=output_shape, cval=-1)
img2_ = warp(img2, (model_robust+offset).inverse, output_shape=output_shape, cval=-1)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3, 2)
f_ax1 = fig.add_subplot(gs[0, :])
plot_matches(f_ax1, img1, img2, keypoints1, keypoints2, matches12)
f_ax1.axis('off')
f_ax1.set_title("Sample ZT13 3-1 vs. Sample ZT13 4-1")
f_ax2 = fig.add_subplot(gs[1, 0])
f_ax2.imshow(img1)
f_ax2.axis('off')
f_ax2.set_title("img1")
f_ax3 = fig.add_subplot(gs[1, 1])
f_ax3.imshow(img1_)
f_ax3.axis('off')
f_ax3.set_title("img1_")
f_ax4 = fig.add_subplot(gs[2, 0])
f_ax4.imshow(img2)
f_ax4.axis('off')
f_ax4.set_title("img2")
f_ax5 = fig.add_subplot(gs[2, 1])
f_ax5.imshow(img2_)
f_ax5.axis('off')
f_ax5.set_title("img2_")
plt.show()


exit()
def add_alpha(image, background=-1):
    rgb = rgb2gray(image)
    alpha = (image != background)
    return np.dstack((image, alpha))

img1_alpha = add_alpha(img1_)
img2_aplha = add_alpha(img2_)

merged = (img1_alpha + img1_alpha)

alpha = merged[..., 3]

merged /= np.maximum(alpha, 1)[..., np.newaxis]