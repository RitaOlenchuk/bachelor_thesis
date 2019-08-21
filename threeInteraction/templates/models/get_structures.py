import numpy as np
import scipy.misc
import matplotlib.pyplot as plt 


image = scipy.misc.imread('/usr/local/hdd/rita/aorta/biorender.png', mode='RGB')

#plt.imshow(image)
#plt.show()

def translate(image, wanted, name):
    image_copy = image.copy()
    if len(wanted)==2:
        white_pixels = np.any(image != wanted[1], axis=-1) * np.any(image != wanted[0], axis=-1)
    else:
        white_pixels = np.any(image != wanted, axis=-1)  
    image_copy[np.invert(white_pixels)] = [0, 0, 0]
    image_copy[white_pixels] = [255, 255, 255]
    scipy.misc.imsave(name, image_copy)

m1 = [236, 161, 160]
m2 = [217, 119, 126]
m3 = [242, 182, 182]
plaque = [230, 134, 18]
phage = [[155, 145, 204], [118, 92, 156]]
neur = [[151, 196, 196],[74, 120, 143]]

translate(image, m1, 'membrane1.png')
translate(image, m2, 'membrane2.png')
translate(image, m3, 'membrane3.png')
translate(image, plaque, 'plaque.png')
translate(image, phage, 'macrophage.png')
translate(image, neur, 'neurophil.png')