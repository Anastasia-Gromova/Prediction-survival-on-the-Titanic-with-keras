import pandas as pd
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
import math

df = pd.read_csv('../datasets/train.csv')
df = df.drop(df.columns[[3, 8]], axis=1)

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
    
data = np.array(df)

#converting cabin types into integers

c = len(df) 
b = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
d = len(b)

for i in range(c):
    a = data[i, 8] != data[i, 8] #checking for nan values
    if a == True:
        e = 0
        e = str(e)
        data[i, 8] = e
    else:
        for j in range(d):
            if b[j] in data[i, 8]:
                e = j + 1
                e = str(e)
                data[i, 8] = e
                
#saving result to a DataFrame  

df.Cabin = data[:,8]

# rescaling Age and Fare columns from 0 to 1

k = 1 / data[:, 4].max()
data[:, 4] = data[:, 4] * k

df.Age = data[:, 4]

k = 1 / data[:, 7].max()
data[:, 7] = data[:, 7] * k

df.Fare = data[:, 7]

# dropping NaNs

df = df.dropna()

# converting to numpy

data = np.array(df)

def train(data, epochs):
    
    a = data.shape[1]-1
    dataY = data[:,1]
    dataX = data[:,2:a]
    
    model = Sequential()
    model.add(Dense(20, input_dim = dataX.shape[1], activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    model.fit(dataX, dataY, validation_split=.33, epochs=epochs, batch_size=32, verbose=2)
    
train(data, 100)
