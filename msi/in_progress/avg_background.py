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

#test python3 avg_background.py --file /usr/local/hdd/rita/msimaging/190927_AR_ZT13_Lipids/190927_AR_ZT13_Lipids.imzML --region 1 --start 0 --save
argument_parser = argparse.ArgumentParser(description='Cluster spectra.')
argument_parser.add_argument("--file", help="specify the name of imzML file", type=str)
argument_parser.add_argument("--region", help="which region would you like to cluster", type=int)
argument_parser.add_argument("--save", help="true if you want to save clustered picture, false otherwise", action='store_true')

args = argument_parser.parse_args()

imzMLfile = args.file
region = args.region
save = args.save

parser = ImzMLParser(imzMLfile)

imze = segment.IMZMLExtract(imzMLfile)

print(imze.get_region_range(region))
xs = (imze.get_region_range(region)[0][0],imze.get_region_range(region)[0][1]+1)
ys = (imze.get_region_range(region)[1][0],imze.get_region_range(region)[1][1]+1)


def tupel2map(spec):
    return dict(zip(spec[0], spec[1]))

mz2intens = {}
print('Calculating pixel map...')
for x in range(xs[0], xs[1]):
    for y in range(ys[0], ys[1]):
        try:
            idx = parser.coordinates.index((x, y, 1))
            sp = tupel2map(parser.getspectrum(idx))
            for k in sp:
                if k in mz2intens:
                    mz2intens[k].append(sp[k])
                else:
                    mz2intens[k] = list()
                    mz2intens[k].append(sp[k])
        except:
            print(f"({x}, {y}, 1) is not in list.")

mz2avg = {}
for key in mz2intens:
    mz2avg[key] = sum(mz2intens[key]) / len(mz2intens[key]) 

if save:
    filename = imzMLfile + "." +str(region)+"_avg"+".pickle"
    dict_file = open(filename,"wb")
    pickle.dump(mz2avg, dict_file)
