import filters
import loader
import numpy as np

FILTERDIM = 9
IMAGEDIM = 45
STEP = 1

class Filter():
    def __init__(self, size, filter):
        self.size = size
        self.filter = filter

    #TODO Implement dynamic stepping
    def convolve(self, imageData, imageSize):
        outputSize = imageSize - self.size
        res = np.array((outputSize, outputSize), dtype = int))
        for x in range(outputSize):
            for y in range(outputSize):
                sum = 0
                for i in range(self.size):
                    for j in range(self.size):
                        if imageData[x + i][y + j]:
                           sum += self.filter[i][j]
                res[x][y] = sum         
        return res

class ConvolveLayer():
    def __init__(self, fileName):
        self.filterList = []
        for filter in filters.FILTERS:
            self.filterList.append(Filter(FILTERDIM, filter)

    def convolve(image):
        output = np.empty(0, dtype=int)
        for filter in self.filterList:
            out = filter.convolve(image, IMAGEDIM)
            np.append(self.output, out.flatten())
        return output
