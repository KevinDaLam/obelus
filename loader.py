import numpy as np

BLACK = "0"
WHITE = "1"
DELIM = " "
DIMENSION = 45
def loadData(fileName):
    imageDict = {} 
    with open(fileName) as file:
        imageName = None
        imageData = np.empty((DIMENSION,DIMENSION), dtype=bool)
        counter = 0
        for line in file:
            if counter == 0:
                imageName = line.strip()
            else:
                imageData[counter - 1] = np.array([x == BLACK for x in line.strip().split(DELIM)])
                if counter == DIMENSION:
                    imageDict[imageName] = imageData
                    imageData = np.empty((DIMENSION,DIMENSION), dtype=bool)
                    counter = -1
            counter += 1
    return imageDict    
