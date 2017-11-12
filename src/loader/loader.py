import numpy as np

BLACK = "0"
WHITE = "1"
DELIM = " "
DIMENSION = 15
def loadData(fileName):
    imageList = [] 
    with open(fileName) as file:
        imageName = None
        imageData = []
        counter = 0
        for line in file:
            if counter == 0:
                imageName = line.strip()
            else:
                imageData.append([x == BLACK for x in line.strip().split(DELIM)])
                if counter == DIMENSION:
                    imageList.append(np.asarray(imageData))
                    imageData = []
                    counter = -1
            counter += 1
    return imageList

def loadBinary(fileName):
    imageList = []
    with open(fileName) as file:
        imageName = None
        imageData = []
        counter = 0
        for line in file:
            if counter == 0:
                imageName = line.strip()
            else:
                for x in line.strip().split(DELIM):
                    imageData.append(int(x))
                if counter == DIMENSION:
                    imageList.append(imageData)
                    imageData = []
                    counter = -1
            counter += 1
    return np.asarray(imageList)
