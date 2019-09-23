import numpy as np
import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def tupel2map(spec):
    return dict(zip(spec[0], spec[1]))

def get_cluster_elements(cluster_id, matrix, parser, xs, ys):
    out = list()
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x,y]==cluster_id:
                out.append(parser.coordinates.index(( xs[0]+y, ys[0]+x, 1)))
    return np.unique(out)

def average_spectra(spectrum1, spectrum2):
    new_spectrum = {}
    for mz in spectrum1.keys():
        new_spectrum[mz] = (spectrum1[mz]+spectrum2[mz])/2
    return new_spectrum

def get_consensus(cluster_id, matrix, dist_dot_product, ids, imzMLfile, xs, ys, plots=False):
    parser = ImzMLParser(imzMLfile)
    
    cluster_ids = get_cluster_elements(cluster_id, matrix, parser, xs, ys)
    cluster_matrix_ids = [ids.index(elem) for elem in cluster_ids]

    distance = np.zeros((len(cluster_matrix_ids), len(cluster_matrix_ids)))

    for i in range(len(cluster_matrix_ids)):
        for j in range(len(cluster_matrix_ids)):
            distance[i, j] = distance[j, i] = dist_dot_product[cluster_matrix_ids[i], cluster_matrix_ids[j]]
    
    np.fill_diagonal(distance, 0)
    Z = linkage(squareform(distance), method = 'average', metric = 'cosine')
    c = fcluster(Z, t=0, criterion='distance')

    order = [x for _, x in sorted(zip(c,range(len(cluster_matrix_ids))), key=lambda pair: pair[0])]

    new_spectum = {}
    spectra_list = list()
    for i in range(len(cluster_matrix_ids)-1):
        if i == 0:
            new_spectum = average_spectra(tupel2map(parser.getspectrum(cluster_matrix_ids[i])), tupel2map(parser.getspectrum(cluster_matrix_ids[i+1])))
        else:
            left = distance[i-1, i]
            right = distance[i, i+1]
            if left > right:
                new_spectum = average_spectra(new_spectum, tupel2map(parser.getspectrum(cluster_matrix_ids[i])))    
            else:
                spectra_list.append(new_spectum)
                new_spectum = average_spectra(tupel2map(parser.getspectrum(cluster_matrix_ids[i])), tupel2map(parser.getspectrum(cluster_matrix_ids[i+1]))) 

    if not spectra_list:
            consensus = spectra_list[0]
            spectra_list.append(new_spectum)
            for spect in spectra_list:
                    consensus = average_spectra(consensus, spect)
    else:
            consensus = new_spectum
    if plots:
        plt.figure()
        for i in cluster_matrix_ids:
                spectrum = tupel2map(parser.getspectrum(i))
                lists = spectrum.items()
                x, y = zip(*lists) # unpack a list of pairs into two tuples
                plt.plot(x,y/max(y), label="Spectral ID {}".format(i))

        lists = consensus.items()
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        plt.plot(x,y/max(y), label="Consensus", c='black')

        plt.xlabel("m/z", fontsize=20)
        plt.ylabel("Intensity (normalized by maximum internsity)", fontsize=20)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    return consensus
