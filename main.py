import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import r2_score

# Read dataset into X and Y
#df = pd.read_csv('C:\\Users\\Chia-Chun\\Desktop\\tensorflow\\train-v3.csv', delim_whitespace=True, header=None)


dx = pd.read_csv('train-v3.csv',header=0)
dataset = dx.values

tX = preprocessing.scale(dataset[:, 2:23])
tY = dataset[:, 1]



dtest = pd.read_csv('test-v3.csv',header=0)
datasettest = dtest.values

ttest =  preprocessing.scale(datasettest[:, 1:22])


#
#valid = pd.read_csv('valid-v3.csv',header=0)
#dataset_valid = valid.values
#
#dataset_valid_tX = preprocessing.scale(dataset_valid[:, 2:18])
#dataset_valid_tY = dataset_valid[:, 1]
#
##print "X: ", X
#print "Y: ", Y


# Define the neural network

from keras.models import Sequential
from keras.layers import Dense

def build_nn():
    model = Sequential()
    model.add(Dense(1024, input_dim=21, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    # No activation needed in output layer (because regression)
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile Model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Evaluate model (kFold cross validation)
from keras.wrappers.scikit_learn import KerasRegressor

# sklearn imports:
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Before feeding the i/p into neural-network, standardise the dataset because all input variables vary in their scales
#estimators 評估

 


estimators = []
estimators.append(('standardise', StandardScaler()))
estimators.append(('multiLayerPerceptron', KerasRegressor(build_fn=build_nn, nb_epoch=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)

log = pipeline.fit(tX, tY)
y_test = pipeline.predict(ttest)


kfold = KFold(n=len(tX), n_folds=10)
results = cross_val_score(pipeline, tX, tY, cv=kfold)

#score = r2_score(tY, pipeline.predict(tX))
#print (score)
#
#score = r2_score(dataset_valid_tY, pipeline.predict(dataset_valid_tX))
#print (score)

print ("Mean: ", results.mean())
print ("StdDev: ", results.std())

#score = r2_score(dataset_valid_tY, pipeline.predict(ttest))

pd.DataFrame(y_test).to_csv("submmit.csv")  