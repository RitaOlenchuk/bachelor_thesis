from pyimzml.ImzMLParser import ImzMLParser

def tupel2map(spec):
    return dict(zip(spec[0], spec[1]))

def get_spec(x, y1, y2, imzML_file):
    parser = ImzMLParser(imzML_file)
    part_map = dict()
    for y in range(y1, y2):
        try:
            idx = parser.coordinates.index((x, y, 1))
            spec_map = tupel2map(parser.getspectrum(idx))
            part_map[idx] = spec_map
        except:
            print(f"({x}, {y}, 1) is not in list.")
    return part_map