import sys
import numpy as np
from PIL import Image
import scipy.misc
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import matplotlib.patches as patches

filename = sys.argv[1]
imagename = sys.argv[2]

tree = ET.parse(filename)
root = tree.getroot()
d = {}
for class_ in root:
    minX = minY = sys.maxsize
    maxX = maxY = -1
    class_name = class_.attrib['Name']
    for element in class_:
        xy = element.attrib['Spot'].split('X')[1].split('Y')
        x = int(xy[0])
        y = int(xy[1])
        if x > maxX:
            maxX = x
        if y > maxY:
            maxY = y
        if x < minX:
            minX = x
        if y < minY:
            minY = y
    d[class_name] = {'minX':minX, 'maxX':maxX, 'minY':minY, 'maxY':maxY}
print(d)

img = Image.open(imagename).convert('L')
img = np.asarray(img)

#img = np.flip(img, 0)
fig,ax = plt.subplots(1)
ax.imshow(img, cmap='gray')
#factor = 1.2625 BCA
factor = 1.37225
xfactor = 1.37225
yfactor = 1.35

xfactor = 0.3025
yfactor = 0.3025

for class_ in d:
    minX = int(d[class_]['minX']/xfactor) 
    maxX = int(d[class_]['maxX']/xfactor) 
    minY = int(d[class_]['minY']/yfactor) 
    maxY = int(d[class_]['maxY']/yfactor)
    rect = patches.Rectangle((minX, minY),maxX-minX, maxY-minY,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    break
plt.show()

#scipy.misc.imsave('window.png', img[minY:maxY, minX:maxX])