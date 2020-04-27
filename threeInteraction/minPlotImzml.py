from pyimzML.ImzMLParser import ImzMLParser, browse, getionimage
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

parser = ImzMLParser("/Users/rita/Uni/bachelor_thesis/190724_BCA_ZT1_Proteins/190724_BCA_ZT1_Proteins_spectra.imzML")

x1 = 328
x2 = 365
y1 = x1
y2 = x2

pixel_map = dict()
for x in range(x1,  x2):
    for y in range(y1, y2):
        try:
            idx = parser.coordinates.index((x, y, 1))
            spec = parser.getspectrum(idx)
            pixel_map[idx] = np.max(spec[1])
        except:
            print(f"({x}, {y}, 1) is not in list.")

ids = list(pixel_map.keys())

image = np.zeros((x2-x1, y2-y1))

for i in ids:
    image[parser.coordinates[i][1]-x1][parser.coordinates[i][0]-y1] = pixel_map[i]
plt.figure(figsize=(20,10))
plt.imshow(image, cmap='hot', interpolation='nearest')
plt.title("Max Intensity")
plt.xticks(range(0,x2-x1,5), np.arange(x1,x2,5))
plt.yticks(range(0,y2-y1,5), np.arange(y1,y2,5))
plt.show()