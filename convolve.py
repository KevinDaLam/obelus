import filters
import matplotlib.pyplot as plt
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
        res = np.empty((outputSize, outputSize), dtype = int)
        for x in range(outputSize):
            for y in range(outputSize):
                sum = 0
                for i in range(self.size):
                    for j in range(self.size):
                        if imageData[x + i, y + j]:
                           sum += self.filter[i][j]
                res[x, y] = sum         
        return res

class ConvolveLayer():
    def __init__(self):
        self.filterList = []
        for filter in filters.FILTERS9X9_2:
            self.filterList.append(Filter(FILTERDIM, filter))

    def convolve(self, image):
        output = np.empty(0, dtype=int)
        for filter in self.filterList:
            out = filter.convolve(image, IMAGEDIM)*75
            plt.imshow(out, cmap="hot", interpolation="nearest")
            plt.show()
            output = np.append(output, out.flatten())
            print out.flatten(), np.size(out)
        return output
