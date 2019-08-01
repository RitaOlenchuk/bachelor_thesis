import sys
import line_spec
import dot_product
import _pickle as pickle
import multiprocessing as mp
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser

imzML_file = sys.argv[1]
x1 = int(sys.argv[2])
x2 = int(sys.argv[3])
y1 = int(sys.argv[4])
y2 = int(sys.argv[5])


###############
import time
start_time = time.time()
###############

#print(mp.cpu_count())

pool = mp.Pool(processes=mp.cpu_count())

result_objects = [pool.apply_async(line_spec.get_spec, args=(x, y1, y2, imzML_file)) for x in range(x1,x2)]

results = [r.get() for r in result_objects]

pool.close()
pool.join()

pixel_map = dict()
for part_map in results:
    pixel_map = {**pixel_map, **part_map}


pool = mp.Pool(processes=mp.cpu_count())
ids = list(pixel_map.keys())
print(ids)
result_objects = [pool.apply_async(dot_product.get_part_m, args=(i, pixel_map)) for i in ids]

results = [r.get() for r in result_objects]

pool.close()
pool.join()

dot_product_similarity = np.zeros((len(ids), len(ids)))
for matr in results:
    dot_product_similarity = np.add(dot_product_similarity, matr)

print(dot_product_similarity.shape)
print(np.max(dot_product_similarity))
print(np.min(dot_product_similarity))
#print(dot_product_similarity)
print("--- %s seconds ---" % (time.time() - start_time))
#print("Saving....")

#spectraMapFile = open("pixel_map.pickle","wb")
#pickle.dump(pixel_map, spectraMapFile)

#print("--- %s seconds ---" % (time.time() - start_time))