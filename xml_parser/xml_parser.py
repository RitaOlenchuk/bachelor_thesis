import sys
import numpy as np
from PIL import Image
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

img = Image.open(imagename)
img = np.asarray(img)
fig,ax = plt.subplots(1)
ax.imshow(img)
print(img.shape)
for class_ in d:
    minX = int(d[class_]['minX'])
    maxX = int(d[class_]['maxX']) 
    minY = int(d[class_]['minY']) 
    maxY = int(d[class_]['maxY']) 
    rect = patches.Rectangle((minX, minY),maxX-minX, maxY-minY,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()