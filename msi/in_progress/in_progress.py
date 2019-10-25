#libraries
import sys, argparse
import _pickle as pickle
from pyimzml.ImzMLParser import ImzMLParser

#scripts
import segment, merge_background, extract_consensus

if __name__ == '__main__':

    #test: python3 in_progress.py --file /usr/local/hdd/rita/msimaging/190927_AR_ZT13_Lipids/190927_AR_ZT13_Lipids.imzML --minus_background 0
    argument_parser = argparse.ArgumentParser(description='Process imzML files to gat image of clustered spectra and respective representative spectra.')
    argument_parser.add_argument("--file", help="specify the name of imzML file", type=str)
    argument_parser.add_argument("--minus_background", help="subtract background spectra (culculated out of smallest region)", action='store_true', required=False)
    argument_parser.add_argument('integers', metavar='N', type=int, nargs='+', help='integer of start position(s)')

    args = argument_parser.parse_args()

    fname = args.file
    minus_background = args.minus_background
    start = args.integers
    start = sorted(start)

    
    for specStart in start:

        imze = segment.IMZMLExtract(fname, specStart=specStart)
        
        if minus_background:
            background_region = imze.get_background()
            back_spectrum = imze.avg_background(background_region)
        else:
            background_region = None
            back_spectrum = None

        for region in imze.get_region_ids():
            if not region==background_region:
                # determine the cosine similarity matrix
                spectra = imze.get_region_array(region, back_spectrum)
                seg = segment.Segmenter()
                outclust = seg.calc_similarity(spectra)

                filename = fname + "." + str(imze.specStart)+"_"+str(region)+".pickle"
                similarity_matrix = open(filename,"wb")
                pickle.dump(outclust, similarity_matrix)

                parser = ImzMLParser(fname)

                xs = (imze.get_region_range(region)[0][0],imze.get_region_range(region)[0][1]+1)
                ys = (imze.get_region_range(region)[1][0],imze.get_region_range(region)[1][1]+1)
                
                #ids - xy coordinates
                id2xy = merge_background.get_id_map(xs, ys, parser)
                ids = list(id2xy.keys())

                # use the similarity matrix for clustering (here: UPGMA)
                clustered_image = merge_background.cluster(outclust, xs, ys, ids, parser, 15, plots=True)

                filename = fname+"."+str(imze.specStart)+"_"+str(region)+"_clustered"+".pickle"
                clustered = open(filename,"wb")
                pickle.dump(clustered_image, clustered)

                #determine consensus sequences for each cluster in clustered image
                cluster2consensus = extract_consensus.get_consensus(parser, clustered_image, outclust, xs, ys, ids, generalplot=True)

                filename = fname+"."+str(imze.specStart)+"_"+str(region)+"_consensus"+".pickle"
                cons = open(filename,"wb")
                pickle.dump(cluster2consensus, cons)
    