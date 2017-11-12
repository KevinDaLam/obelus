from kokoro import kokoro
from convolve import convolve
from loader import loader
import numpy as np
import datetime

files = ("data/shrink_0.txt", "data/shrink_1.txt")
testfiles = ("data/test_0.txt", "data/test_1.txt", "data/test_2.txt")
featLength = 676

def main():
    print(datetime.datetime.now())
    CLayer = convolve.ConvolveLayer()
    input_data = np.empty((0, featLength), dtype = int)
    output_data = []
    
    #for index, ofile in enumerate(files):
    #     load = loader.loadBinary(ofile)
    #     print load.shape[0]
    #     print load.shape[1]
    #     for i in range(load.shape[0]):
    #         output_data.append([index])
    #     input_data = np.vstack((input_data, load)) 
    #output_data = np.asarray(output_data)
    #print input_data

    for index, ofile in enumerate(files):
        print("Loading {}".format(ofile))
        imageList = loader.loadData(ofile) 
        #outputArr = [index]
        outputArr = [0]*len(files)
        outputArr[index] = 1
        print("Loaded: {} images".format(len(imageList)))
        for image in imageList:
            input_data = np.vstack((input_data, CLayer.convolve(image))) 
            output_data.append(outputArr) 
    output_data = np.asmatrix(output_data)

    ANN = kokoro.ANNetwork(0.5, featLength, 300, 2, 2, 1)   
    ANN.Train(input_data, output_data, 1)
    
    #test_data = loader.loadBinary("test_shrink.txt")
    #for i in range(test_data.shape[0]):
    #    print(ANN.Predict(np.asmatrix(test_data[i])))

    with open("output.csv", 'w') as outfile:
        for ofiles in testfiles:
            outfile.write("{}\n".format(ofiles))
            imageList = loader.loadData(ofiles)
            for image in imageList:
                test_data = CLayer.convolve(image)
                output = ANN.Predict(np.asmatrix(test_data))
                outfile.write(str(output.tolist()))
                outfile.write("\n")
    print(datetime.datetime.now())
if __name__ == "__main__":
    main()
