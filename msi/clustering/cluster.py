import numpy as np
import sys
import segment
import matplotlib.pyplot as plt
import _pickle as pickle
from Bio import Cluster
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage


imzMLfile = sys.argv[1]
region = int(sys.argv[2])
start = int(sys.argv[3])
print(imzMLfile)
parser = ImzMLParser(imzMLfile)

similarity_matrix = open(imzMLfile+"."+str(start)+"_"+str(region)+".pickle","rb")
similarity = pickle.load(similarity_matrix)

def get_matrix(m, xm, ym):
    result = np.zeros((m.shape))
    for i in range(ym):
        for j in range(xm): #for each block
            base_x = i*xm
            base_y = j*ym

            new_x = i*xm + j
            for ii in range(xm):
                for jj in range(ym): #for each element in the block
                    new_y = ii*ym + jj
                    baseXII = base_x+ii
                    baseYJJ = base_y+jj

                    oldValue = m[baseXII, baseYJJ]

                    result[new_x, new_y] = oldValue

                
    return result

imze = segment.IMZMLExtract(imzMLfile)

print(imze.get_region_range(region))
xs = (imze.get_region_range(region)[0][0],imze.get_region_range(region)[0][1]+1)
ys = (imze.get_region_range(region)[1][0],imze.get_region_range(region)[1][1]+1)

dist_dot_product = 1 - similarity
dist_pixel = np.zeros((dist_dot_product.shape[0], dist_dot_product.shape[1]))

xy2id = {}
print('Calculating pixel map...')
for x in range(xs[0], xs[1]):
    for y in range(ys[0], ys[1]):
        try:
            idx = parser.coordinates.index((x, y, 1))
            xy2id[idx] = (x, y)
        except:
            print(f"({x}, {y}, 1) is not in list.")

ids = list(xy2id.keys())
print('Calculating real distances...')
for i in range(len(ids)):
    coordI = np.asarray(parser.coordinates[ids[i]])
    for j in range(i, len(ids)):
        coordJ = np.asarray(parser.coordinates[ids[j]])
        dist = np.linalg.norm(coordI-coordJ)
        dist_pixel[i, j] = dist_pixel[j, i] = dist

normed_dist_pixel = dist_pixel / np.max(dist_pixel)
log_dist_dot = np.log(dist_dot_product+1)
log_dist_dot = log_dist_dot / np.max(log_dist_dot)

general_dot_product = 0.95*log_dist_dot + 0.05*normed_dist_pixel

np.fill_diagonal(dist_dot_product, 0)
np.fill_diagonal(general_dot_product, 0)
print('UPGMA clustering...')

Z1 = linkage(squareform(dist_dot_product), method = 'average', metric = 'cosine')
Z2 = linkage(squareform(general_dot_product), method = 'average', metric = 'cosine')

#c1 = fcluster(Z1, t=0.1, criterion='distance')
#c2 = fcluster(Z2, t=0.2, criterion='distance')
c1 = fcluster(Z1, t=7, criterion='maxclust')
c2 = fcluster(Z2, t=7, criterion='maxclust')
image_UPGMA_dot = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))
image_UPGMA_pixel = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

print('Transforming in real coordinates...')
for i in ids:
    image_UPGMA_dot[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c1[ids.index(i)]
    image_UPGMA_pixel[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c2[ids.index(i)]

print("Kmeans clustering...")
kmeans1 = KMeans(n_clusters=5, random_state=0).fit(dist_dot_product)
kmeans2 = KMeans(n_clusters=5, random_state=0).fit(general_dot_product)
image_kmeans_dot = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))
image_kmeans_pixel = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))
print('Transforming in real coordinates...')
for i in ids:
    image_kmeans_dot[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = kmeans1.labels_[ids.index(i)]
    image_kmeans_pixel[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = kmeans2.labels_[ids.index(i)]


print('Kmedoid clustering...')
kmedoid_dot = Cluster.kmedoids(dist_dot_product, nclusters=5, npass=5, initialid=None) #Distance Matrix!!!
kmedoid_pixel = Cluster.kmedoids(general_dot_product, nclusters=5, npass=5, initialid=None) #Distance Matrix!!!
image_kmedoid_dot = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))
image_kmedoid_pixel = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))
for i in ids:
    image_kmedoid_dot[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = kmedoid_dot[0][ids.index(i)]
    image_kmedoid_pixel[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = kmedoid_pixel[0][ids.index(i)]

fig = plt.figure(figsize=(20, 30))
fig.add_subplot(3,2,1)
im = plt.imshow(image_kmedoid_dot, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('K medoid distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(3,2,2)
im = plt.imshow(image_kmedoid_pixel, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('K medoid dot-product and pixel distances')
plt.colorbar(im, fraction=0.046, pad=0.04)


fig.add_subplot(3,2,3)
im = plt.imshow(image_kmeans_dot, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('K means distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(3,2,4)
im = plt.imshow(image_kmeans_pixel, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('K means dot-product and pixel distances')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(3,2,5)
im = plt.imshow(image_UPGMA_dot, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(3,2,6)
im = plt.imshow(image_UPGMA_pixel, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA + pixel distances')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.show()