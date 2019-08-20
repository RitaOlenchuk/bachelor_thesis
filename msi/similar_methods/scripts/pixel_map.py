import sys
import line_spec
import dot_product
import _pickle as pickle
import multiprocessing as mp
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from itertools import product

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

a = np.array(list(pixel_map.keys())) 
id_pairs = list(set(list(product(a,a))))

pool = mp.Pool(processes=mp.cpu_count())
result_objects = [pool.apply_async(dot_product.get_similarity, args=(t[0], t[1], pixel_map[t[0]], pixel_map[t[1]])) for t in id_pairs]
results = [r.get() for r in result_objects]

pool.close()
pool.join()

ids = list(pixel_map.keys())
d = { ids[i]:i for i in range(len(ids)) }

dot_product_similarity = np.zeros((len(ids), len(ids)))
for r in results:
    dot_product_similarity[d[r[0]], d[r[1]]] = dot_product_similarity[d[r[1]], d[r[0]]] = r[2]
print("--- %s seconds ---" % (time.time() - start_time))
import matplotlib.pyplot as plt
plt.imshow(dot_product_similarity)
plt.show()

exit()
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