import numpy as np

def readFile(filepath, deli):
    data = []
    f = open(filepath, 'r')
    for line in f:
        partialData = line.split(deli)
        data.append([int(z) for z in partialData[:3]])
    f.close()
    data = np.array(data)

    return data