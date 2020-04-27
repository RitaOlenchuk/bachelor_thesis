import numpy as np
import imageio
import matplotlib.pyplot as plt 
from skimage import io, color

#image = imageio.imread('/Users/rita/Uni/bachelor_thesis/git/bachelor_thesis/threeInteraction/templates/models/base_model.png')
image = color.rgba2rgb (io.imread('/Users/rita/Uni/bachelor_thesis/git/bachelor_thesis/threeInteraction/templates/models/base_model.png'))
image = image * 255
image = image.astype(int)

def translate(image, wanted, name):
    image_copy = image.copy()
    if len(wanted)==2:
        white_pixels = np.any(image != wanted[1], axis=-1) * np.any(image != wanted[0], axis=-1)
    else:
        white_pixels = np.any(image != wanted, axis=-1)  
    image_copy[np.invert(white_pixels)] = [0, 0, 0]
    image_copy[white_pixels] = [1, 1, 1]
    #io.imsave(name, image_copy)

m1 = [201, 201, 201]
m2 = [166, 166, 166]
m3 = [199, 199, 199]
plaque = [240, 240, 240]
phage = [23, 23, 23]
#neur = [167, 88, 150]

translate(image, m1, 'membrane1.png')
translate(image, m2, 'membrane2.png')
translate(image, m3, 'membrane3.png')
translate(image, plaque, 'plaque.png')
translate(image, phage, 'macrophage.png')
#translate(image, neur, 'neurophil.png')