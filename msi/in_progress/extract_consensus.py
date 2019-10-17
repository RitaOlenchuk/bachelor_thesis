import numpy as np
import csv
import sys
import segment
import consensus
import argparse
import imageio
import dabest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.metrics.pairwise import cosine_similarity
from pyimzml.ImzMLParser import ImzMLParser

#test python3 extract_consensus.py --file /usr/local/hdd/rita/msimaging/181114_AT1_Slide_D_Proteins.imzML --region 0 --start 0 --generalplot --boxplot --mass 12345
argument_parser = argparse.ArgumentParser(description='Cluster spectra.')
argument_parser.add_argument("--file", help="specify the name of imzML file")
argument_parser.add_argument("--region", help="which region would you like to cluster", type=int)
argument_parser.add_argument("--start", help="start of the spectra", type=int)
argument_parser.add_argument("--generalplot", help="true if you want to plot the general plot with all consensus spectra, false otherwise", action='store_true', required=False)
argument_parser.add_argument("--boxplot", help="true if you want to plot the boxplot of similarity of consensus spectr within each cluster, false otherwise", action='store_true', required=False)
argument_parser.add_argument("--mass", help="if mass given, you will plot the mean diff between the distribution of this mass in cluster 0 with compaired to all other", type=int, required=False)
argument_parser.add_argument("--textconsensus", help="true if you want to write consensus dict into .txt", action='store_true', required=False)

args = argument_parser.parse_args()

imzMLfile = args.file
region = args.region
start = args.start 
generalplot = args.generalplot
boxplot = args.boxplot
textconsensus = args.textconsensus
if args.mass:
    massplot = True
    MASS = args.mass
else:
    massplot = False

parser = ImzMLParser(imzMLfile)

cluster_file = open(imzMLfile+"."+str(start)+"_"+str(region)+"_clustered"+".pickle","rb")
cluster_image = pickle.load(cluster_file)

similarity_matrix = open(imzMLfile+"."+str(start)+"_"+str(region)+".pickle","rb")
similarity = pickle.load(similarity_matrix)
dist_dot_product = 1 - similarity

imze = segment.IMZMLExtract(imzMLfile)

print(imze.get_region_range(region))
xs = (imze.get_region_range(region)[0][0],imze.get_region_range(region)[0][1]+1)
ys = (imze.get_region_range(region)[1][0],imze.get_region_range(region)[1][1]+1)

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

def get_similarity(map_1, map_2):
    #Similarity (dot product) of two comlpete spectra
    intens1 = np.array(list(map_1.values()))
    intens2 = np.array(list(map_2.values()))
    
    intens1 = intens1.reshape(1, len(intens1))
    intens2 = intens2.reshape(1, len(intens2))
    cos_lib = cosine_similarity(intens1, intens2)

    return cos_lib[0][0]

cluster2concensus = {}
cluster2comparison = {}
cluster2MASS = {}
for cluster in np.unique(cluster_image):
        print(cluster)
        if (generalplot or boxplot):
            #array = consensus.get_cluster_elements(cluster, cluster_image, parser, xs, ys)
            #cluster2concensus[cluster] = consensus.tupel2map(parser.getspectrum(array[0]))
            cluster2concensus[cluster] = consensus.get_average_consensus(cluster, cluster_image, imzMLfile, xs, ys)
        cluster_ids = consensus.get_cluster_elements(cluster, cluster_image, parser, xs, ys)
        tmp = list()
        for i in cluster_ids:
            if massplot:
                spec = consensus.tupel2map(parser.getspectrum(i))
                new_spec = {round(k):v for k, v in spec.items()}
                curMASS = None
                for key in list(new_spec.keys()):
                        if curMASS == None or abs(key - MASS) < abs(curMASS - MASS):
                                curMASS = key

                MASS=curMASS

                mass = new_spec[MASS]
                if cluster in cluster2MASS:
                        cluster2MASS[cluster].append(mass)
                else:
                        cluster2MASS[cluster] = list()
                        cluster2MASS[cluster].append(mass)
            if generalplot or boxplot:
                tmp.append(1-(get_similarity(cluster2concensus[cluster], consensus.tupel2map(parser.getspectrum(i)))))
        if boxplot:
            cluster2comparison[cluster] = tmp

#Massplot comparison is plotted here
if massplot:
    with open(str(MASS)+'_mass.csv', mode='w') as mass_file:
        mass_writer = csv.writer(mass_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        mass_writer.writerow(list(cluster2MASS.keys()))
        max_length = 0
        for cluster in cluster2MASS:
            if len(cluster2MASS[cluster])>max_length:
                    max_length = len(cluster2MASS[cluster])
        for i in range(max_length):
            tmp = np.full(len(list(cluster2MASS.keys())), np.nan)
            for cluster in cluster2MASS:
                    if len(cluster2MASS[cluster]) > i:
                            tmp[cluster] = cluster2MASS[cluster][i]
            mass_writer.writerow(tmp)

    # Load the iris dataset. Requires internet access.
    mass = pd.read_csv(str(MASS)+'_mass.csv')
    mass = mass.rename(columns={"0": "Cluster 0", "1": "Cluster 1", "2": "Cluster 2", "3": "Cluster 3", "4": "Cluster 4", "5": "Cluster 5"})
    #mass['Cluster 0'] = 0
    # Load the above data into `dabest`.
    shared_control = dabest.load(mass, idx=("Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"))
    # Produce a Cumming estimation plot.
    shared_control.mean_diff.plot()
    plt.show()

#General plot is plotted here
if generalplot:
    consensus_distance = np.zeros((len(cluster2concensus.keys()), len(cluster2concensus.keys())))
    for cluster1 in range(len(cluster2concensus.keys())):
            for cluster2 in range(cluster1, len(cluster2concensus.keys())):
                    consensus_distance[cluster1, cluster2] = consensus_distance[cluster2, cluster1] = 1 - get_similarity(cluster2concensus[cluster1], cluster2concensus[cluster2])

    fig = plt.figure()
    grid = plt.GridSpec(len(np.unique(cluster_image)), 3, wspace=0.1, hspace=0.35)

    for i in cluster2concensus.keys():
            i = int(i)
            ax = fig.add_subplot(grid[i, 1:])
            lists = cluster2concensus[i].items()
            x, y = zip(*lists) # unpack a list of pairs into two tuples
            ax.plot(x,y/max(y), label="Consensus cluster {}".format(i))

            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity(nAU)")
            ax.legend()

    ax = fig.add_subplot(grid[:int(len(np.unique(cluster_image))/2), :1])
    im = ax.imshow(cluster_image, cmap='gist_ncar', interpolation='nearest')

    ax.set_yticks(range(0,ys[1]-ys[0],5))
    ax.set_yticklabels(np.arange(ys[0], ys[1],5))
    ax.set_xticks(range(0,xs[1]-xs[0],5))
    ax.set_xticklabels(np.arange(xs[0], xs[1],5))
    ax.set_title('UPGMA + pixel distances')# t='+str(0.19))
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(grid[int(len(np.unique(cluster_image))/2):, :1])
    im = ax.imshow(consensus_distance, interpolation='nearest', cmap='hot')
    ax.set_yticks(range(0,len(cluster2concensus.keys())))
    ax.set_xticks(range(0,len(cluster2concensus.keys())))
    ax.set_title('Distance matrix')
    fig.colorbar(im, ax=ax)

    plt.show()

#Boxpplot is plotted here    
if boxplot:
    sns.set(style="ticks")
    data =  list()
    for i in list(cluster2comparison.keys()):
        data.append(cluster2comparison[i])

    f, ax = plt.subplots()
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=data)

    # Add in points to show each observation
    sns.swarmplot(data=data)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set_title('Distances of cluster elements to cluster representative')
    ax.set_ylabel('Distance')
    #ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    plt.show()

if textconsensus:
    file = open("consensus.txt","w")
    for i in cluster2concensus.keys():
        file.write("{}\t{}\n".format(int(i), cluster2concensus[i]))
    file.close() 