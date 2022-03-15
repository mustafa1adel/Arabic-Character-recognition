import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# re-read the data
trainX = pd.read_csv('Datasets/csvTrainImages 13440x1024.csv', header=None)
trainY = pd.read_csv('Datasets/csvTrainLabel 13440x1.csv', header=None)
# reading training sets
testX = pd.read_csv('Datasets/csvTestImages 3360x1024.csv', header=None)
testY = pd.read_csv('Datasets/csvTestLabel 3360x1.csv', header=None)

# reshape features
trainX = np.array(trainX).reshape(trainX.shape[0], 32, 32 )
testX = np.array(testX).reshape(testX.shape[0], 32, 32)

# normalize the images
trainX = trainX / trainX.max()
testX = testX / testX.max()

# create the encoder
encoder = LabelEncoder()
encoder.fit(trainY)
# encode the labels
trainY = encoder.transform(trainY)
testY = encoder.transform(testY)


def load_data(model_type = None):
    global trainX, testX, trainY, testY
    
    if model_type.upper() == "CNN":
        # reshape the data for Convolutional layers
        trainX = trainX.reshape([-1, 32, 32, 1])
        testX = testX.reshape([-1, 32, 32, 1])
        return trainX, testX, trainY, testY
    
    return trainX, testX, trainY, testY
