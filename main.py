import kokoro
import convolve
import loader
import numpy as np
import datetime

files = ("0_shrink.txt", "1_shrink.txt")
featLength = 225

def main():
    print datetime.datetime.now()
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

    for index, file in enumerate(files):
        print "Loading {}".format(file)
        imageList = loader.loadData(file) 
        outputArr = [index]
        print "Loaded: {} images".format(len(imageList))
        for image in imageList:
            input_data = np.vstack((input_data, CLayer.convolve(image))) 
            output_data.append(outputArr) 
    output_data = np.asarray(output_data)

    ANN = kokoro.ANNetwork(1, featLength, featLength + 1, 1, 1, 1)   
    ANN.Train(input_data, output_data, 10)
    
    #test_data = loader.loadBinary("test_shrink.txt")
    #for i in range(test_data.shape[0]):
    #    print(ANN.Predict(np.asmatrix(test_data[i])))

    imageList = loader.loadData("test_shrink.txt")
    for image in imageList:
        test_data = CLayer.convolve(image)
        print(ANN.Predict(np.asmatrix(test_data)))

    print datetime.datetime.now()
if __name__ == "__main__":
    main()
