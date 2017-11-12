from . import filters
import matplotlib.pyplot as plt
import numpy as np

FILTERDIM = 3
IMAGEDIM = 15
STEP = 1
THRESH = 2
class Filter():
    def __init__(self, size, filter):
        self.size = size
        self.filter = filter

    #TODO Implement dynamic stepping
    def convolve(self, imageData, imageSize):
        outputSize = imageSize - self.size + 1
        res = np.empty((outputSize, outputSize), dtype = int)
        for x in range(outputSize):
            for y in range(outputSize):
                sum = 0
                for i in range(self.size):
                    for j in range(self.size):
                        if imageData[x + i, y + j]:
                           sum += self.filter[i][j]
                if sum > THRESH: 
                    res[x, y] = 1         
                else:
                    res[x, y] = -1
        return res

class ConvolveLayer():
    def __init__(self):
        self.filterList = []
        for filter in filters.FILTERS3X3:
            self.filterList.append(Filter(FILTERDIM, filter))

    def convolve(self, image):
        output = np.empty(0, dtype=int)
        for filter in self.filterList:
            out = filter.convolve(image, IMAGEDIM)
            #plt.imshow(out, cmap="hot", interpolation="nearest")
            #plt.show()
            output = np.append(output, out.flatten())
        normed = (output - output.mean())/output.std()
        return normed
