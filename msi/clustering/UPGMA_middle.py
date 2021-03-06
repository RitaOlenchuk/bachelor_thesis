import numpy as np
import sys
import segment
import consensus
import seaborn as sns
import matplotlib.pyplot as plt
import _pickle as pickle
from Bio import Cluster
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage

#test: python3 cluster.py /Users/rita/Uni/bachelor_thesis/msi/181114_AT1_Slide_D_Proteins.imzML 0 0
imzMLfile = sys.argv[1]
region = int(sys.argv[2])
start = int(sys.argv[3])
print(imzMLfile)
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

np.fill_diagonal(dist_dot_product, 0)
np.fill_diagonal(general_dot_product, 0)
print('UPGMA clustering...')

Z1 = linkage(squareform(dist_dot_product), method = 'average', metric = 'cosine')
Z2 = linkage(squareform(general_dot_product), method = 'average', metric = 'cosine')

c1 = fcluster(Z1, t=0.1, criterion='distance')
c2 = fcluster(Z2, t=0.2, criterion='distance')
image_UPGMA_dot = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))
image_UPGMA_pixel = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

c11 = fcluster(Z1, t=0.08, criterion='distance')
c22 = fcluster(Z2, t=0.19, criterion='distance')
image_UPGMA_dot1 = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))
image_UPGMA_pixel1 = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

print('Transforming in real coordinates...')
for i in ids:
    image_UPGMA_dot[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c1[ids.index(i)]
    image_UPGMA_pixel[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c2[ids.index(i)]

    image_UPGMA_dot1[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c11[ids.index(i)]
    image_UPGMA_pixel1[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c22[ids.index(i)]


'''
fig = plt.figure(figsize=(50, 20))

fig.add_subplot(2,4,1)
im = plt.imshow(image_UPGMA_dot, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only t='+str(0.1))
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(2,4,5)
im = plt.imshow(image_UPGMA_pixel, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA + pixel distances t='+str(0.2))
plt.colorbar(im, fraction=0.046, pad=0.04)
'''
image_UPGMA_dot[(image_UPGMA_dot == 5) | (image_UPGMA_dot == 3) ] = 0
image_UPGMA_pixel[(image_UPGMA_pixel == 3)] = 0

dict_dot = dict(zip(list(np.unique(image_UPGMA_dot)), list(range(len(np.unique(image_UPGMA_dot))))))
dict_pixel = dict(zip(list(np.unique(image_UPGMA_pixel)), list(range(len(np.unique(image_UPGMA_pixel))))))

for i in range(image_UPGMA_dot.shape[0]):
    for j in range(image_UPGMA_dot.shape[1]):
        image_UPGMA_dot[i,j] = dict_dot[image_UPGMA_dot[i,j]]
        image_UPGMA_pixel[i,j] = dict_pixel[image_UPGMA_pixel[i,j]]
'''
fig.add_subplot(2,4,2)
im = plt.imshow(image_UPGMA_dot)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(2,4,6)
im = plt.imshow(image_UPGMA_pixel)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA + pixel distances')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(2,4,3)
im = plt.imshow(image_UPGMA_dot1, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(2,4,7)
im = plt.imshow(image_UPGMA_pixel1, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA + pixel distances')
plt.colorbar(im, fraction=0.046, pad=0.04)
'''
image_UPGMA_dot[image_UPGMA_dot > 0] = 1
image_UPGMA_pixel[image_UPGMA_pixel > 0] = 1

image_UPGMA_dot1 = np.multiply(image_UPGMA_dot1, image_UPGMA_dot)
image_UPGMA_pixel1 = np.multiply(image_UPGMA_pixel1, image_UPGMA_pixel)

dict_dot1 = dict(zip(list(np.unique(image_UPGMA_dot1)), list(range(len(np.unique(image_UPGMA_dot1))))))
dict_pixel1 = dict(zip(list(np.unique(image_UPGMA_pixel1)), list(range(len(np.unique(image_UPGMA_pixel1))))))

for i in range(image_UPGMA_dot1.shape[0]):
    for j in range(image_UPGMA_dot1.shape[1]):
        image_UPGMA_dot1[i,j] = dict_dot1[image_UPGMA_dot1[i,j]]
        image_UPGMA_pixel1[i,j] = dict_pixel1[image_UPGMA_pixel1[i,j]]
'''
fig.add_subplot(2,4,4)
im = plt.imshow(image_UPGMA_dot1)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(2,4,8)
im = plt.imshow(image_UPGMA_pixel1)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA + pixel distances')
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.show()
'''
def get_similarity(map_1, map_2):
    #Similarity (dot product) of two comlpete spectra
    intens1 = np.array(list(map_1.values()))
    intens2 = np.array(list(map_2.values()))
    
    intens1 = intens1.reshape(1, len(intens1))
    intens2 = intens2.reshape(1, len(intens2))
    cos_lib = cosine_similarity(intens1, intens2)

    return cos_lib[0][0]

print(len(np.unique(image_UPGMA_pixel1)))
cluster2concensus = {}
cluster2comparison = {}
for cluster in np.unique(image_UPGMA_pixel1):
        print(cluster)
        cluster2concensus[cluster] = consensus.get_consensus(cluster, image_UPGMA_pixel1, dist_dot_product, ids, imzMLfile, xs, ys)
        cluster_ids = consensus.get_cluster_elements(cluster, image_UPGMA_pixel1, parser, xs, ys)
        tmp = list()
        for i in cluster_ids:
                tmp.append(1-(get_similarity(cluster2concensus[cluster], consensus.tupel2map(parser.getspectrum(i)))))
        cluster2comparison[cluster] = tmp


consensus_distance = np.zeros((len(cluster2concensus.keys()), len(cluster2concensus.keys())))
for cluster1 in range(len(cluster2concensus.keys())):
        for cluster2 in range(cluster1, len(cluster2concensus.keys())):
                consensus_distance[cluster1, cluster2] = consensus_distance[cluster2, cluster1] = 1 - get_similarity(cluster2concensus[cluster1], cluster2concensus[cluster2])

fig = plt.figure()
grid = plt.GridSpec(len(np.unique(image_UPGMA_pixel1)), 3, wspace=0.1, hspace=0.1)

for i in cluster2concensus.keys():
        i = int(i)
        ax = fig.add_subplot(grid[i, 1:])
        lists = cluster2concensus[i].items()
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        ax.plot(x,y/max(y), label="Consensus cluster {}".format(i))

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity (normalized by maximum internsity)")
        ax.legend()

ax = fig.add_subplot(grid[:int(len(np.unique(image_UPGMA_pixel1))/2), :1])
im = ax.imshow(image_UPGMA_pixel1, cmap='hot', interpolation='nearest')

ax.set_yticks(range(0,ys[1]-ys[0],5))
ax.set_yticklabels(np.arange(ys[0], ys[1],5))
ax.set_xticks(range(0,xs[1]-xs[0],5))
ax.set_xticklabels(np.arange(xs[0], xs[1],5))
ax.set_title('UPGMA + pixel distances t='+str(0.19))
fig.colorbar(im, ax=ax)

ax = fig.add_subplot(grid[int(len(np.unique(image_UPGMA_pixel1))/2):, :1])
im = ax.imshow(consensus_distance, interpolation='nearest', cmap='cool')
ax.set_yticks(range(0,len(cluster2concensus.keys())))
ax.set_xticks(range(0,len(cluster2concensus.keys())))
ax.set_title('Distance matrix')
fig.colorbar(im, ax=ax)

plt.show()

sns.set(style="ticks")
data =  list()
for i in range(len(cluster2comparison.keys())):
        data.append(cluster2comparison[i])

f, ax = plt.subplots()
# Plot the orbital period with horizontal boxes
sns.boxplot(data=data)

# Add in points to show each observation
sns.swarmplot(data=data)

# Tweak the visual presentation
ax.xaxis.grid(True)
#ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.show()