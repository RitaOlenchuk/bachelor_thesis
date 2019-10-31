import imreg_dft
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('/usr/local/hdd/aorta_images/HE_prediction/11_outfile.png').convert('L')
img = np.array(img)
img[img==255] = 0
img = Image.fromarray(img)
img.save('11_outfile.png')

img = Image.open('/usr/local/hdd/aorta_images/HE_prediction/2_outfile.png').convert('L')
img = np.array(img)
img[img==255] = 0
img = Image.fromarray(img)
img.save('2_outfile.png')


moving = plt.imread('11_outfile.png')
fixed = plt.imread('2_outfile.png')

moving = np.array(moving)
moving = moving/np.max(moving)
#moving[(moving < 1) & (moving > 0)] = 1
moving = moving.astype(int)

fixed = np.array(fixed)
fixed = fixed/np.max(fixed)
#fixed[(fixed < 1) & (fixed > 0)] = 1
fixed = fixed.astype(int)


#translation = imreg_dft.imreg.translation(np.array(fixed), np.array(moving))
#transformedImage = imreg_dft.imreg.transform_img(np.array(moving), scale=1.0, angle=translation['angle'], tvec=translation['tvec'], mode='constant', bgval=None, order=1)
similarity = imreg_dft.imreg.similarity(fixed, moving)
transformedImage = imreg_dft.imreg.transform_img(moving, scale=similarity['scale'], angle=similarity['angle'], tvec=similarity['tvec'], mode='constant', bgval=None, order=1)
imreg_dft.imreg.imshow(fixed, moving, transformedImage)# similarity['timg'])
plt.show()
original_moving = plt.imread('/usr/local/hdd/aorta_images/HE_prediction/11_infile.png')
original_moving = np.array(original_moving)

original_fixed = plt.imread('/usr/local/hdd/aorta_images/HE_prediction/2_infile.png')
original_fixed = np.array(original_fixed)

transformedImage = imreg_dft.imreg.transform_img(original_moving, scale=similarity['scale'], angle=similarity['angle'], tvec=similarity['tvec'], mode='constant', bgval=None, order=1)
imreg_dft.imreg.imshow(original_fixed, moving, original_moving)

plt.show()