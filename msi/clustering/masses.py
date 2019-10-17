import sys
import segment
import numpy as np
import pytraj as pt
import _pickle as pickle
import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser

'''
cluster_map = open("/usr/local/hdd/rita/CLUSTERING/pySRM/src/pySRM/pysrm/ImportantMassFile_2.pickle","rb")
peak_freq = pickle.load(cluster_map)
print(len(list(peak_freq.keys())))
for peak in list(peak_freq.keys()):
    if peak_freq[peak] > 2000:
        print('ha')
        peak_freq.pop(peak, None)
print(len(list(peak_freq.keys())))
#print(peak_freq.keys())
exit()
'''
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

def tupel2map(spec):
    return dict(zip(spec[0], spec[1]))

def get_peaks(spec):
    interval = 100#len(spec.keys())//1000
       
    peaks = set()
    
    for intens in pt.tools.n_grams(list(spec.keys()), interval):
        maxI = 0
        maxMZ = 0
        
        epshull = (max(intens)-min(intens))/2
        
        for mz in intens:
            if spec[mz] > maxI:
                maxI = spec[mz]
                maxMZ = mz
        
        tmp = maxMZ
        
        addPeak = True
        if len(peaks) > 0:
            
            #exist already registered peak within epsilon hull with lower intensity?
            for p in peaks:
                
                if abs(p-tmp) < epshull:
                    if spec[p] < spec[tmp]:
                        peaks.remove(p)
                        peaks.add(tmp)
                        addPeak = False
                        break
                    else:
                        
                        addPeak = False
                        break
                        
        if addPeak:
            
            allValues = [spec[mz] for mz in intens]
            if maxI > 5*np.median(allValues):
                peaks.add(tmp)
            
    return np.array(list(peaks))

count = 0
peak_freq = {}
xy2id = {}
plt.figure()
for x in range(xs[0], xs[1]):
    for y in range(ys[0], ys[1]):
        print(count)
        count += 1
        try:
            idx = parser.coordinates.index((x, y, 1))
            xy2id[idx] = (x, y)
            spec = tupel2map(parser.getspectrum(idx))
            peaks = get_peaks(spec)
            for peak in peaks:
                #peak = int(peak)
                peak = round(peak, 2)
                if peak in peak_freq:
                    peak_freq[peak] += 1
                else:
                    peak_freq[peak] = 1 
        except:
            print(f"({x}, {y}, 1) is not in list.")

all_sp = len(list(xy2id.keys()))
print('Saving...')
print('Total {}'.format(all_sp))
mapFile = open("ImportantMassFile_2.pickle","wb")
pickle.dump(peak_freq, mapFile)