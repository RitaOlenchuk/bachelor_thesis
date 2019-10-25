#libraries
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pyimzml.ImzMLParser import ImzMLParser
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import sys

#scripts
import segment

def cluster(similarity_matrix, xs, ys, ids, parser, N=15, plots=False):
    dist_dot_product = 1 - similarity_matrix

    #determine NORMALIZED euclid distance between the coordinates
    dist_pixel = get_euclid_distance(ids, parser, np.zeros((dist_dot_product.shape[0], dist_dot_product.shape[1])))

    #NORMALIZED log of distance matrix values
    log_dist_dot = np.log(dist_dot_product+1)
    log_dist_dot = log_dist_dot / np.max(log_dist_dot)

    #final matrix for clustering
    general_dot_product = 0.95*log_dist_dot + 0.05*dist_pixel

    #in case of some roand inaccuracy
    np.fill_diagonal(general_dot_product, 0)

    #UPGMA with N clusters
    clustered_image = upgma(general_dot_product, N, xs, ys, ids, parser)

    #merge background clusters
    generalized_clustered_image = merge_edge_clusters(xs, ys, ids, clustered_image, N, dist_dot_product, parser, plots)

    return generalized_clustered_image

def get_id_map(xs, ys, parser):
    id2xy = {}
    print('Calculating pixel map...')
    for x in range(xs[0], xs[1]):
        for y in range(ys[0], ys[1]):
            try:
                idx = parser.coordinates.index((x, y, 1))
                id2xy[idx] = (x, y)
            except ValueError:
                print(f"({x}, {y}, 1) is not in list.")
    return id2xy

def get_euclid_distance(ids, parser, dist_pixel):
    print('Calculating real distances...')
    for i in range(len(ids)):
        coordI = np.asarray(parser.coordinates[ids[i]])
        for j in range(i, len(ids)):
            coordJ = np.asarray(parser.coordinates[ids[j]])
            dist = np.linalg.norm(coordI-coordJ)
            dist_pixel[i, j] = dist_pixel[j, i] = dist
    normed_dist_pixel = dist_pixel / np.max(dist_pixel)
    return normed_dist_pixel

def upgma(dist_matrix, N, xs, ys, ids, parser):
    print('UPGMA clustering...')
    Z = linkage(squareform(dist_matrix), method = 'average', metric = 'cosine')
    c = fcluster(Z, t=N, criterion='maxclust')

    image_UPGMA = np.zeros((ys[1]-ys[0], xs[1]-xs[0]), dtype=int)

    print('Transforming in real coordinates...')
    for i in ids:
        image_UPGMA[parser.coordinates[i][1]-ys[0]][parser.coordinates[i][0]-xs[1]] = c[ids.index(i)]
    return image_UPGMA

def get_edge_clusters(xs, ys, image):
    edge_ids = list()
    for x in [xs[0], xs[0]+1, xs[0]+2, xs[1]-2, xs[1]-1]:
        for y in [ys[0], ys[0]+1, ys[0]+2, ys[1]-2, ys[1]-1]:
            try:
                edge_ids.append(image[y-ys[0], x-xs[0]])
            except:
                print(f"({x}, {y}, 1) is not in list.")

    return np.unique(edge_ids)

def merge_edge_clusters(xs, ys, ids, image, N, dist_dot_product, parser, plots):
    edge_ids = get_edge_clusters(xs, ys, image)

    cluster_pair2dist = defaultdict()
    edge_pair2dist = defaultdict()
    unique_clusters = np.unique(image)

    cluster2ids = defaultdict()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            try:
                if not image[x, y] in cluster2ids:
                    cluster2ids[image[x,y]] = list()
                cluster2ids[image[x,y]].append(parser.coordinates.index(( xs[0]+y, ys[0]+x, 1)))
            except:
                print(f"({x}, {y}, 1) is not in list.")

    for cluster1 in range(len(unique_clusters)):
        print("Cluster1: "+str(unique_clusters[cluster1]))
        cluster1_ids = cluster2ids[unique_clusters[cluster1]]
        for cluster2 in range(cluster1, len(unique_clusters)):
            print(unique_clusters[cluster2])
            if not unique_clusters[cluster1] == unique_clusters[cluster2]:
                cluster2_ids = cluster2ids[unique_clusters[cluster2]]
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
    
    threshold = 0.0
    if not edge_pair2dist:
        threshold = 0.85
    else:
        for pair in edge_pair2dist:
            threshold += np.mean(edge_pair2dist[pair])
        threshold /= len(list(edge_pair2dist.keys()))
        print("threshold = "+str(threshold))
        if threshold < 0.85:
            threshold = 0.85
    print("threshold = "+str(threshold))

    new_image = np.copy(image)

    for pair in cluster_pair2dist:
        pair_mean = np.mean(cluster_pair2dist[pair])
        if pair_mean >= threshold:
            cluster1 = int(pair.split('vs')[0])
            cluster2 = int(pair.split('vs')[1])
            new_image[new_image==cluster2] = cluster1

    dict_pixel = dict(zip(list(np.unique(new_image)), list(range(len(np.unique(new_image))))))

    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i,j] = dict_pixel[new_image[i,j]]

    if plots:
        plot_overview(xs, ys, N, image, new_image, cluster_pair2dist)
    
    return new_image

def plot_overview(xs, ys, N, image, new_image, cluster_pair2dist):
    fig = plt.figure()
    grid = plt.GridSpec(1, 2, wspace=0.1, hspace=0.35)
    ax = fig.add_subplot(grid[0, :1])
    im = plt.imshow(image, cmap='hot', interpolation='nearest')
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