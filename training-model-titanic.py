import pandas as pd
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
import math

def clean(df):
    df = df.drop(['Name', 'Ticket'], axis=1)

    l=[]

    #converting column with genders to 0 and 1

    s = "female"
    df.Sex = df.Sex == s
    df.Sex = df.Sex.astype(int)

    # filling NaNs of the Age column

    age = int(df.Age.mean())
    std = int(df.Age.std())
    lo = int(age - std)
    hi = int(age + std)

    c = len(df.Age)

    for i in range(c):

        if np.isnan(df.Age[i]) == True:

            l.append(random.randint(lo,hi))

        else:
            l.append(df.Age[i])

    df.Age = l

    #converting embarked types into integers 0, 1 and 2

    mapping = [('C','0'), ('S','1'), ('Q','2')]
    for k,v in mapping:
        df.Embarked = df.Embarked.replace(k,v)

    #converting cabin types into integers
    
    temp_data = np.array(df.Cabin)

    c = len(df) 
    b = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    d = len(b)

    for i in range(c):
        a = temp_data[i] != temp_data[i] #checking for nan values
        if a == True:
            e = 0
            e = str(e)
            temp_data[i] = e
        else:
            for j in range(d):
                if b[j] in temp_data[i]:
                    e = j + 1
                    e = str(e)
                    temp_data[i] = e

    #saving result to a DataFrame  

    df.Cabin = temp_data[:]
    
    # dealing with NaN

    temp = df.Fare.tolist()
    for i in range(len(df)):
        if temp[i] != temp[i]:
            temp[i] = 0
    df.Fare = temp

    # rescaling Age and Fare columns from 0 to 1
    
    temp_data = np.array(df.Age)

    k = 1 / temp_data[:].max()
    temp_data[:] = temp_data[:] * k

    df.Age = temp_data[:]
    
    temp_data = np.array(df.Fare)

    k = 1 / temp_data[:].max()
    
    temp_data[:] = temp_data[:] * k

    df.Fare = temp_data[:]

    # converting to numpy

    data = np.array(df)

    return data

def train(train_data, epochs, test_data, val=.33):
    
    a = data.shape[1]-1
    dataY = data[:,1]
    dataX = data[:,2:]
    
    model = Sequential()
    model.add(Dense(20, input_dim = dataX.shape[1], activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    model.fit(dataX, dataY, validation_split=val, epochs=epochs, batch_size=32, verbose=2)
    
    preds = model.predict(test_data)
    
    return preds

train_data = clean(pd.read_csv('../datasets/train.csv'))
test_data = clean(pd.read_csv('../datasets/test.csv'))

predictions = train(train_data, 50, test_data[:,1:], val=0)

df = pd.DataFrame()
df['PassengerId'] = pd.read_csv('../datasets/test.csv').PassengerId
df['Survived'] = np.array(predictions)

df.Survived = df.Survived >= .5
df.Survived = df.Survived.astype(int)

df.to_csv('titanic_submission.csv', index=False)
