import numpy as np

def readFile(filepath):
    data = []
    f = open(filepath, 'r')
    for line in f:
        partialData = line.split('::')
        data.append([int(z) for z in partialData[:3]])
    f.close()
    data = np.array(data)

    return data