import numpy as np
import sys
import segment
import argparse
import consensus
import matplotlib.pyplot as plt
import _pickle as pickle
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pyimzml.ImzMLParser import ImzMLParser

N = 15

#test python3 merge_background.py --file /usr/local/hdd/rita/msimaging/181114_AT1_Slide_D_Proteins.imzML --region 0 --start 0
argument_parser = argparse.ArgumentParser(description='Cluster spectra.')
argument_parser.add_argument("--file", help="specify the name of imzML file", type=str)
argument_parser.add_argument("--region", help="which region would you like to cluster", type=int)
argument_parser.add_argument("--start", help="start of the spectra", type=int)

args = argument_parser.parse_args()

imzMLfile = args.file
region = args.region
start = args.start 

parser = ImzMLParser(imzMLfile)

similarity_matrix = open(imzMLfile+"."+str(start)+"_"+str(region)+".pickle","rb")
similarity = pickle.load(similarity_matrix)

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

np.fill_diagonal(general_dot_product, 0)

print('UPGMA clustering...')
Z = linkage(squareform(general_dot_product), method = 'average', metric = 'cosine')
c = fcluster(Z, t=N, criterion='maxclust')

image_UPGMA = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

image_UPGMA = image_UPGMA.astype(int)
print('Transforming in real coordinates...')
for i in ids:
    image_UPGMA[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c[ids.index(i)]
print(image_UPGMA.shape)

edge_ids = list()
for x in [xs[0], xs[0]+1, xs[0]+2, xs[1]-2, xs[1]-1]:
    for y in [ys[0], ys[0]+1, ys[0]+2, ys[1]-2, ys[1]-1]:
        try:
            edge_ids.append(image_UPGMA[y-ys[0], x-xs[0]])
        except:
            print(f"({x}, {y}, 1) is not in list.")

edge_ids = np.unique(edge_ids)
print(edge_ids)

cluster_pair2dist = defaultdict()
edge_pair2dist = defaultdict()
unique_clusters = np.unique(image_UPGMA)

frequency = defaultdict()

for x in range(image_UPGMA.shape[0]):
    for y in range(image_UPGMA.shape[1]):
        try:
            if not image_UPGMA[x, y] in frequency:
                frequency[image_UPGMA[x,y]] = list()
            frequency[image_UPGMA[x,y]].append(parser.coordinates.index(( xs[0]+y, ys[0]+x, 1)))
        except:
            print(f"({x}, {y}, 1) is not in list.")

for cluster1 in range(len(unique_clusters)):
    print("Cluster1: "+str(unique_clusters[cluster1]))
    cluster1_ids = frequency[unique_clusters[cluster1]]
    for cluster2 in range(cluster1, len(unique_clusters)):
        print(unique_clusters[cluster2])
        if not unique_clusters[cluster1] == unique_clusters[cluster2]:
            cluster2_ids = frequency[unique_clusters[cluster2]]
            for i in range(len(cluster1_ids)):
                for j in range(len(cluster2_ids)):
                    dist = 1-dist_dot_product[ids.index(cluster1_ids[i]), ids.index(cluster2_ids[j])]
                    key = str(unique_clusters[cluster1])+"vs"+str(unique_clusters[cluster2])
                    if unique_clusters[cluster1] in edge_ids and unique_clusters[cluster2] in edge_ids:
                        if not key in edge_pair2dist:
                            edge_pair2dist[key] = list()
                        edge_pair2dist[key].append(dist)
                    if not key in cluster_pair2dist:
                        cluster_pair2dist[key] = list()
                    cluster_pair2dist[key].append(dist)

t = 0.0
if not edge_pair2dist:
    t = 0.9
else:
    for pair in edge_pair2dist:
        t += np.mean(edge_pair2dist[pair])
    t /= len(list(edge_pair2dist.keys()))
    print("t="+str(t))
    if t < 0.9:
        t = 0.9

new_image = np.copy(image_UPGMA)
print("t="+str(t))
for pair in cluster_pair2dist:
    pair_mean = np.mean(cluster_pair2dist[pair])
    if pair_mean >= t:
        cluster1 = int(pair.split('vs')[0])
        cluster2 = int(pair.split('vs')[1])
        new_image[new_image==cluster2] = cluster1

dict_pixel = dict(zip(list(np.unique(new_image)), list(range(len(np.unique(new_image))))))

for i in range(new_image.shape[0]):
    for j in range(new_image.shape[1]):
        new_image[i,j] = dict_pixel[new_image[i,j]]

fig = plt.figure()
grid = plt.GridSpec(1, 2, wspace=0.1, hspace=0.35)
ax = fig.add_subplot(grid[0, :1])
im = plt.imshow(image_UPGMA, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA N='+str(N))
fig.colorbar(im, ax=ax)

ax = fig.add_subplot(grid[0, 1:])
im = plt.imshow(new_image, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('Updated')
fig.colorbar(im, ax=ax)
plt.show()

fig = plt.figure()

labels, data = cluster_pair2dist.keys(), cluster_pair2dist.values()
plt.boxplot(data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show()

filename = imzMLfile+"."+str(start)+"_"+str(region)+"_clustered"+".pickle"
clustered = open(filename,"wb")
pickle.dump(new_image, clustered)