import numpy as np
from scipy import misc
import ctypes
import argparse
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from collections import defaultdict
from pyimzml.ImzMLParser import ImzMLParser, browse, getionimage
import logging
import _pickle as pickle
import math

baseFolder = str(os.path.dirname(os.path.realpath(__file__)))

lib = ctypes.cdll.LoadLibrary(baseFolder+'/../../cppSRM/lib/libSRM.so')

class Segmenter():

    def __init__(self):
        lib.StatisticalRegionMerging_New.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
        lib.StatisticalRegionMerging_New.restype = ctypes.c_void_p

        lib.SRM_processFloat.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_processFloat.restype = ctypes.POINTER(ctypes.c_uint32)

        lib.SRM_calc_similarity.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_calc_similarity.restype = ctypes.POINTER(ctypes.c_float)

        lib.SRM_test_matrix.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.SRM_test_matrix.restype = None

        lib.StatisticalRegionMerging_mode_dot.argtypes = []
        lib.StatisticalRegionMerging_mode_dot.restype = None

    def calc_similarity(self, inputarray):

        #load image
        dims = 1

        inputarray = inputarray.astype(np.float32)
        
        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]

        qs = []
        qArr = (ctypes.c_float * len(qs))(*qs)

        logger = logging.getLogger('dev')
        logger.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        logger.addHandler(consoleHandler)

        formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
        consoleHandler.setFormatter(formatter)

        logger.info('information message')

        logger.info("Creating C++ obj")

        print("dimensions", dims)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))

        image_p = inputarray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        logger.info("Starting calc similarity c++")
        retValues = lib.SRM_calc_similarity(self.obj, inputarray.shape[0], inputarray.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(inputarray.shape[0]*inputarray.shape[1], inputarray.shape[0]*inputarray.shape[1]))

        print(outclust.dtype)
        print(outclust.shape)

        logger.info("displaying matrix")

        return outclust



    def segment_array(self, inputarray, qs=[256, 0.5, 0.25], imagedim = None):

        dims = 1

        if len(inputarray.shape) > 2:
            dims = inputarray.shape[2]

        qArr = (ctypes.c_float * len(qs))(*qs)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        
        dimage = (inputarray / np.max(inputarray)) * 255
        image_p = dimage.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        retValues = lib.SRM_processFloat(self.obj, dimage.shape[0], dimage.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(len(qs), dimage.shape[0], dimage.shape[1]))

        print(outclust.dtype)
        print(outclust.shape)

        outdict = {}

        for i,q in enumerate(qs):
            outdict[q] = outclust[i, :,:]

        if imagedim == None:
            imagedim = int(dims/3)

        image = inputarray[:,:,imagedim]
        image = image / np.max(image)

        return image, outdict



    def segment_image(self, imagepath, qs=[256, 0.5, 0.25]):

        #load image
        image = plt.imread(imagepath)
        image = image.astype(np.float32)
        image = image / np.max(image)

        print(image.shape)
        print(image.dtype)
        print(np.min(image), np.max(image))

        dims = 1

        if len(image.shape) > 2:
            dims = image.shape[2]

        qArr = (ctypes.c_float * len(qs))(*qs)

        self.obj = lib.StatisticalRegionMerging_New(dims, qArr, len(qs))
        
        dimage = image * 255
        image_p = dimage.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        retValues = lib.SRM_processFloat(self.obj, dimage.shape[0], dimage.shape[1], image_p)

        outclust = np.ctypeslib.as_array(retValues, shape=(len(qs), dimage.shape[0], dimage.shape[1]))

        print(outclust.dtype)
        print(outclust.shape)

        outdict = {}

        for i,q in enumerate(qs):
            outdict[q] = outclust[i, :,:]

        return image, outdict


class IMZMLExtract:

    def __init__(self, fname, specStart=0):
        #fname = "/mnt/d/dev/data/190724_AR_ZT1_Proteins/190724_AR_ZT1_Proteins_spectra.imzML"

        self.fname = fname
        self.parser = ImzMLParser(fname)
        self.dregions = None

        self.mzValues = self.parser.getspectrum(0)[0]

        self.specStart = specStart

        if self.specStart != 0:
            self.mzValues = self.mzValues[self.specStart:]
            print("WARNING: SPECTRA STARTING AT POSITION", self.specStart)

        self.find_regions()

    def get_region_ids(self):
        return [x for x in self.dregions]

    def get_spectrum(self, specid):
        spectra1 = self.parser.getspectrum(specid)[1]
        return spectra1

    def compare_spectra(self, specid1, specid2):

        spectra1 = self.parser.getspectrum(specid1)[1]
        spectra2 = self.parser.getspectrum(specid2)[1]

        ssum = 0.0
        len1 = 0.0
        len2 = 0.0

        assert(len(spectra1) == len(spectra2))

        for i in range(0, len(spectra1)):

            ssum += spectra1[i] * spectra2[i]
            len1 += spectra1[i]*spectra1[i]
            len2 += spectra2[i]*spectra2[i]

        len1 = math.sqrt(len1)
        len2 = math.sqrt(len2)

        return ssum/(len1*len2)


    def get_mz_index(self, value):

        curIdxDist = 1000000
        curIdx = 0

        for idx, x in enumerate(self.mzValues):
            dist = abs(x-value)

            if dist < curIdxDist:
                curIdx = idx
                curIdxDist = dist
            
        return curIdx

    def get_region_spectra(self, regionid, back_spectrum = None):

        if not regionid in self.dregions:
            return None
        
        outspectra = {}

        for coord in self.dregions[regionid]:

            spectID = self.parser.coordinates.index(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            cspec = self.parser.getspectrum( spectID )[1]
            cspec = cspec[self.specStart:]
            
            if len(cspec)==0:
                print("0 spec")
                continue
            if back_spectrum:
                cspec = np.subtract(cspec, back_spectrum)
                cspec[cspec < 0.0] = 0.0
                cspec = cspec - np.min(cspec)
            cspec = cspec/np.max(cspec)
            outspectra[coord] = cspec

        return outspectra

    def get_region_range(self, regionid):

        allpixels = self.dregions[regionid]

        minx = min([x[0] for x in allpixels])
        maxx = max([x[0] for x in allpixels])

        miny = min([x[1] for x in allpixels])
        maxy = max([x[1] for x in allpixels])

        minz = min([x[2] for x in allpixels])
        maxz = max([x[2] for x in allpixels])

        spectraLength = 0
        for coord in self.dregions[regionid]:

            spectID = self.parser.coordinates.index(coord)

            if spectID == None or spectID < 0:
                print("Invalid coordinate", coord)
                continue

            splen = self.parser.mzLengths[spectID]-self.specStart

            spectraLength = max(spectraLength, splen)

        return (minx, maxx), (miny, maxy), (minz, maxz), spectraLength

    def get_region_shape(self, regionid):

        rr = self.get_region_range(regionid)
        xr,yr,zr,sc = rr

        imzeShape = [
            xr[1]-xr[0]+1,
            yr[1]-yr[0]+1
        ]

        if zr[1]-zr[0]+1 > 1:
            imzeShape.append( zr[1]-zr[0]+1 )

        imzeShape.append(sc)

        spectraShape = tuple(imzeShape)

        return spectraShape


    def get_region_array(self, regionid, back_spectrum = None):

        xr,yr,zr,sc = self.get_region_range(regionid)
        rs = self.get_region_shape(regionid)
        print(rs)

        sarray = np.zeros( rs, dtype=np.float32 )

        coord2spec = self.get_region_spectra(regionid, back_spectrum)

        for coord in coord2spec:
            xpos = coord[0]-xr[0]
            ypos = coord[1]-yr[0]

            spectra = coord2spec[coord]

            if len(spectra) < sc:
                spectra = np.pad(spectra, ((0,0),(0, sc-len(spectra) )), mode='constant', constant_values=0)

            sarray[xpos, ypos, :] = spectra

        return sarray

    def find_regions(self):

        if os.path.isfile(self.fname + ".regions"):

            print("Opening regions file for", self.fname)

            with open(self.fname + ".regions", 'r') as fin:
                self.dregions = defaultdict(list)

                for line in fin:
                    line = line.strip().split("\t")

                    coords = [int(x) for x in line]

                    self.dregions[coords[3]].append( tuple(coords[0:3]) )

            for regionid in self.dregions:

                allpixels = self.dregions[regionid]

                minx = min([x[0] for x in allpixels])
                maxx = max([x[0] for x in allpixels])

                miny = min([x[1] for x in allpixels])
                maxy = max([x[1] for x in allpixels])


        else:

            self.dregions = self.__detectRegions(self.parser.coordinates)

            with open(self.fname + ".regions", 'w') as outfn:

                for regionid in self.dregions:

                    for pixel in self.dregions[regionid]:

                        print("\t".join([str(x) for x in pixel]), regionid, sep="\t", file=outfn)
    
    
    def __dist(self, x,y):

        assert(len(x)==len(y))

        dist = 0
        for pidx in range(0, len(x)):

            dist += abs(x[pidx]-y[pidx])

        return dist


    def __detectRegions(self, allpixels):

        allregions = []

        for idx, pixel in enumerate(allpixels):

            if len(allregions) == 0:
                allregions.append([pixel])
                continue

            if idx % 1000 == 0:
                print("At pixel", idx , "of", len(allpixels), "with", len(allregions), "regions")


            accRegions = []

            for ridx, region in enumerate(allregions):

                for coord in region:
                    if self.__dist(coord, pixel) <= 1:
                        accRegions.append(ridx)
                        break


            if len(accRegions) == 0:
                allregions.append([pixel])

            elif len(accRegions) == 1:

                for ridx in accRegions:
                    allregions[ridx].append(pixel)

            elif len(accRegions) > 1:

                bc = len(allregions)

                totalRegion = []
                for ridx in accRegions:
                    totalRegion += allregions[ridx]

                for ridx in sorted(accRegions, reverse=True):
                    del allregions[ridx]

                allregions.append(totalRegion)

                ac = len(allregions)

                assert(ac == bc + 1 - len(accRegions))

        outregions = {}

        for i in range(0, len(allregions)):
            outregions[i] = [tuple(x) for x in allregions[i]]

        return outregions

    def avg_background(self, background_id):
        xs = (self.get_region_range(background_id)[0][0],self.get_region_range(background_id)[0][1])
        ys = (self.get_region_range(background_id)[1][0],self.get_region_range(background_id)[1][1])

        mz2intens = {}
        
        for x in range(xs[0], xs[1]):
            for y in range(ys[0], ys[1]):
                try:
                    idx = self.parser.coordinates.index((x, y, 1))
                    tupl = self.parser.getspectrum(idx)
                    mz = tupl[0]
                    inten = tupl[1]
                    for i in range(len(mz)):
                        if mz[i] in mz2intens:
                            mz2intens[mz[i]].append(inten[i])
                        else:
                            mz2intens[mz[i]] = list()
                            mz2intens[mz[i]].append(inten[i])
                except:
                    print(f"({x}, {y}, 1) is not in list.")

        mz2avg = {}
        for key in mz2intens:
            mz2avg[key] = sum(mz2intens[key]) / len(mz2intens[key]) 
        return list(mz2avg.values())


    def get_background(self):
        # determine the smallest region in imzML file which will be considered as background measurement
        max_surface = sys.maxsize
        background_region = None
        for region in self.get_region_ids():
                xs = (self.get_region_range(region)[0][0],self.get_region_range(region)[0][1])
                ys = (self.get_region_range(region)[1][0],self.get_region_range(region)[1][1])
                surface = (xs[1] - xs[0])*(ys[1] - ys[0])
                
                if max_surface > surface:
                    max_surface = surface
                    background_region = region
        print("Background is: "+str(background_region))
        return background_region