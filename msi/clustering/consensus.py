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

#test: python3 consensus.py /Users/rita/Uni/bachelor_thesis/msi/181114_AT1_Slide_D_Proteins.imzML 0 0
imzMLfile = sys.argv[1]
region = int(sys.argv[2])
start = int(sys.argv[3])

parser = ImzMLParser(imzMLfile)

similarity_matrix = open(imzMLfile+"."+str(start)+"_"+str(region)+".pickle","rb")
similarity = pickle.load(similarity_matrix)

imze = segment.IMZMLExtract(imzMLfile)

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

c = fcluster(Z, t=0.2, criterion='distance')
image_UPGMA = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

def get_cluster_elements(cluster_id, matrix):
    out = list()
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x,y]==cluster_id:
                out.append(parser.coordinates.index(( xs[0]+y, ys[0]+x, 1)))
    return np.unique(out)

def tupel2map(spec):
    return dict(zip(spec[0], spec[1]))

def get_consensus(id1, id2):
    spectrum1 = tupel2map(parser.getspectrum(id1))
    spectrum2 = tupel2map(parser.getspectrum(id2))

    new_spectrum = {}

    for mz in spectrum1.keys():
        new_spectrum[mz] = (spectrum1[mz]+spectrum2[mz])/2

    plt.figure()
    lists = spectrum1.items()
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x,y/max(y), label="Spectral ID {}".format(id1))

    lists = spectrum2.items()
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x,y/max(y), label="Spectral ID {}".format(id2))

    lists = new_spectrum.items()
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x,y/max(y), label="New Spectrum")

    plt.title("Distance {}".format(dist_dot_product[id1, id2]), fontsize=30)
    plt.xlabel("m/z", fontsize=20)
    plt.ylabel("Intensity (normalized by maximum internsity)", fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    return new_spectrum


c2 = fcluster(Z, t=0, criterion='distance')
image_UPGMA2 = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

print('Transforming in real coordinates...')
for i in ids:
    image_UPGMA[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c[ids.index(i)]
    image_UPGMA2[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c2[ids.index(i)]

print(get_cluster_elements(1, image_UPGMA))

cluster_1 = get_cluster_elements(1, image_UPGMA)

print([ids.index(elem) for elem in cluster_1])

real_cluster_1 = [ids.index(elem) for elem in cluster_1]

get_consensus(real_cluster_1[0], real_cluster_1[1])


exit()



fig = plt.figure()

fig.add_subplot(1,4,1)
im = plt.imshow(image_UPGMA, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA t='+str(0.1))
plt.colorbar(im, fraction=0.046, pad=0.04)

image_UPGMA[(image_UPGMA == 3)] = 0

dict_pixel = dict(zip(list(np.unique(image_UPGMA)), list(range(len(np.unique(image_UPGMA))))))

for i in range(image_UPGMA.shape[0]):
    for j in range(image_UPGMA.shape[1]):
        image_UPGMA[i,j] = dict_pixel[image_UPGMA[i,j]]

fig.add_subplot(1,4,2)
im = plt.imshow(image_UPGMA)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(1,4,3)
im = plt.imshow(image_UPGMA2, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

image_UPGMA[image_UPGMA > 0] = 1

image_UPGMA2= np.multiply(image_UPGMA2, image_UPGMA)

dict_pixel2 = dict(zip(list(np.unique(image_UPGMA2)), list(range(len(np.unique(image_UPGMA2))))))

for i in range(image_UPGMA2.shape[0]):
    for j in range(image_UPGMA2.shape[1]):
        image_UPGMA2[i,j] = dict_pixel2[image_UPGMA2[i,j]]

fig.add_subplot(1,4,4)
im = plt.imshow(image_UPGMA2)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.show()