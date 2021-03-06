import numpy as np
from PIL import Image
import imageio
from matplotlib import pyplot as plt

img_path = '/Users/rita/Uni/bachelor_thesis/DL/2_outfile.png'
img = Image.open(img_path)
img = np.array(img, dtype=np.float32)
img = img / np.max(np.max(img))
print(set(img.flatten()))
img [img < 0.3] = 0
selParts = (img > 0.3) & (img < 0.5)
img [img >= 0.5] = 1
img [selParts] = 2
img [img >= 3] = 0
print(set(img.flatten()))
background = np.argwhere(img==0)
membrane = np.argwhere(img==1)
plaque = np.argwhere(img==2)


img_path2 = '/Users/rita/Uni/bachelor_thesis/DL/segmented2.png'
img2 = Image.open(img_path2)
img2 = np.array(img2)
n_segments = (len(set(img2.flatten())))
print('Number of segments: ',n_segments)

fig = plt.figure()

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('DL', fontsize=16)

plt.subplot(222)
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.title('Segmented', fontsize=16)

plt.show()

#fig = plt.figure()

#plt.subplot(221)
#plt.imshow(img2, cmap='gray')
#plt.axis('off')
#plt.title('DL', fontsize=16)

#m = 38
#tmp = np.argwhere(img2==m)
#plt.subplot(222)
#plt.imshow(img2, cmap='gray')
#plt.scatter([t1[1] for t1 in tmp], [t2[0] for t2 in tmp], c='red', s=1, label='Segment '+str(m))
#plt.axis('off')
#plt.legend()
#plt.title('Segmented', fontsize=16)

#plt.show()

output_image = np.zeros(img.shape)
for segm in range(n_segments):
    
    coords = np.argwhere(img2 == segm) #all coordinates that belong to the segment
    frequencies = [0, 0, 0] #how often the coordinate in selected segment belong to (background, membrane, plaque)
    
    for coord in coords:
        if img[coord[0], coord[1]] == 0:
            frequencies[0] += 1
        elif img[coord[0], coord[1]] == 1:
            frequencies[1] += 1
        elif img[coord[0], coord[1]] == 2:
            frequencies[2] += 1
    struct = np.argmax(frequencies)
    for c in coords:
        output_image[c[0], c[1]] = struct
    #output_image[coords] = struct

fig = plt.figure()
fig.suptitle('Q = 0.002', fontsize=16)

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.title('DL', fontsize=16)

plt.subplot(132)
plt.imshow(img2)
#plt.scatter([t1[1] for t1 in tmp], [t2[0] for t2 in tmp], c='red', s=1, label='Segment '+str(m))
plt.axis('off')
plt.title('Segmented', fontsize=16)

plt.subplot(133)
plt.imshow(output_image)
plt.axis('off')
plt.title('Mix', fontsize=16)
plt.savefig('dl_segm_mixed2.png')
plt.show()



