import numpy as np
import csv
import sys
import segment
import argparse
import consensus
import seaborn as sns
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pyimzml.ImzMLParser import ImzMLParser

#test python3 cluster_background.py --file /usr/local/hdd/rita/msimaging/181114_AT1_Slide_D_Proteins.imzML --region 0 --start 0 --generaln 0.2 --finaln 0.19 --save 3 7 8 9
argument_parser = argparse.ArgumentParser(description='Cluster spectra.')
argument_parser.add_argument("--file", help="specify the name of imzML file", type=str)
argument_parser.add_argument("--region", help="which region would you like to cluster", type=int)
argument_parser.add_argument("--start", help="start of the spectra", type=int)
argument_parser.add_argument("--generaln", help="specify the number of clusters to extract only the regions not included the background", type=float)
argument_parser.add_argument("--finaln", help="specify the number of clusters to extract the regions in aorta", type=float)
argument_parser.add_argument("--save", help="true if you want to save clustered picture, false otherwise", action='store_true')
argument_parser.add_argument('integers', metavar='N', type=int, nargs='+', help='integer of cluster ids that should be part of the background')

args = argument_parser.parse_args()

imzMLfile = args.file
region = args.region
start = args.start 
t1 = args.generaln 
t2 = args.finaln
background_clusters = args.integers
background_clusters = sorted(background_clusters)
save = args.save

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

Z = linkage(squareform(general_dot_product), method = 'average', metric = 'cosine')

c = fcluster(Z, t=t1, criterion='maxclust')
#c = fcluster(Z, t=t1, criterion='distance')
image_UPGMA = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

c2 = fcluster(Z, t=t2, criterion='maxclust')
#c2 = fcluster(Z, t=t2, criterion='distance')
image_UPGMA2 = np.zeros((ys[1]-ys[0], xs[1]-xs[0]))

print('Transforming in real coordinates...')
for i in ids:
    image_UPGMA[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c[ids.index(i)]

    image_UPGMA2[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c2[ids.index(i)]

fig = plt.figure(figsize=(50, 20))

fig.add_subplot(1,4,1)
im = plt.imshow(image_UPGMA, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only t='+str(t1))
plt.colorbar(im, fraction=0.046, pad=0.04)


if save:
    filename = imzMLfile+"."+str(start)+"_"+str(region)+"_clustered"+".pickle"
    clustered = open(filename,"wb")
    pickle.dump(image_UPGMA, clustered)

for elem in background_clusters:
    if elem in image_UPGMA:
        image_UPGMA[(image_UPGMA == elem)] = 0

dict_pixel = dict(zip(list(np.unique(image_UPGMA)), list(range(len(np.unique(image_UPGMA))))))

for i in range(image_UPGMA.shape[0]):
    for j in range(image_UPGMA.shape[1]):
        image_UPGMA[i,j] = dict_pixel[image_UPGMA[i,j]]

fig.add_subplot(1,4,2)
im = plt.imshow(image_UPGMA)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.add_subplot(1,4,3)
im = plt.imshow(image_UPGMA2, cmap='hot', interpolation='nearest')
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only t='+str(t2))
plt.colorbar(im, fraction=0.046, pad=0.04)

image_UPGMA[image_UPGMA > 0] = 1

image_UPGMA2 = np.multiply(image_UPGMA2, image_UPGMA)

dict_pixel2 = dict(zip(list(np.unique(image_UPGMA2)), list(range(len(np.unique(image_UPGMA2))))))

for i in range(image_UPGMA.shape[0]):
    for j in range(image_UPGMA.shape[1]):
        image_UPGMA2[i,j] = dict_pixel2[image_UPGMA2[i,j]]

fig.add_subplot(1,4,4)
im = plt.imshow(image_UPGMA2)
plt.xticks(range(0,xs[1]-xs[0],5), np.arange(xs[0], xs[1],5))
plt.yticks(range(0,ys[1]-ys[0],5), np.arange(ys[0], ys[1],5))
plt.title('UPGMA distance only')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.show()