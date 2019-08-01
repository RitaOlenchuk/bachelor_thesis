import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(spec1, spec2, id_map):
    map_1 = id_map[spec1]
    map_2 = id_map[spec2]
    
    #Similarity (dot product) of two comlpete spectra
    intens1 = np.array(list(map_1.values()))
    intens2 = np.array(list(map_2.values()))
    
    intens1 = intens1/np.max(intens1)
    intens2 = intens2/np.max(intens2)
    
    intens1 = intens1.reshape(1, len(intens1))
    intens2 = intens2.reshape(1, len(intens2))
    cos_lib = cosine_similarity(intens1, intens2)

    return cos_lib[0][0]


def get_part_m(row_id, pixel_map):
    print('here')
    ids = list(pixel_map.keys())
    dot_product_part = np.zeros((len(ids), len(ids)))
    start_element = ids.index(row_id)
    print(start_element)
    for i in range(start_element, len(ids)):
        dot_product_part[i, start_element] = dot_product_part[start_element, i] = get_similarity(row_id, ids[i], pixel_map)
    return dot_product_part


