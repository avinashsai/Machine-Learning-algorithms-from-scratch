import os
import re
import math
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


data = pd.read_csv('diabetes.csv',sep=',')

print(data.head(3))

labels = (data["Outcome"]==1).astype(int)

data = data.drop(["Outcome"],axis=1)

d1,d2,y1,y2 = train_test_split(data,labels,test_size=0.3,random_state=42)

x_train = d1.as_matrix()

x_test = d2.as_matrix()

y_train = y1.as_matrix()

y_test = y2.as_matrix()

train_zeros = sum(y_train==0)
train_ones = sum(y_train==1)

test_zeros = sum(y_test==0)
test_ones = sum(y_test==1)

train_length = len(x_train)
test_length = len(x_test)

print(x_train.shape)
print(x_test.shape)

features = x_train.shape[1]
print(features)

x_train_zeros = np.zeros((train_zeros,features))
x_train_ones = np.zeros((train_ones,features))

x_test_zeros = np.zeros((test_zeros,features))
x_test_ones = np.zeros((test_ones,features))

def make_features(values,labels,ones,zeros):
    j = 0
    k = 0
    for i in range(len(labels)):
        if(labels[i]==0):
            zeros[j]=values[i]
            j = j + 1
        else:
            ones[k]=values[i]
            k = k + 1
    return ones,zeros


x_train_ones,x_train_zeros = make_features(x_train,y_train,x_train_ones,x_train_zeros)
x_test_ones,x_test_zeros = make_features(x_test,y_test,x_test_ones,x_test_zeros)



def cal_scores(col):
    return np.mean(col),np.std(col)



x_train_ones_scores = np.zeros((features,2))
x_train_zeros_scores = np.zeros((features,2))
x_test_ones_scores = np.zeros((features,2))
x_test_zeros_scores = np.zeros((features,2))



for i in range(features):
    x_train_ones_scores[i] = cal_scores(x_train_ones[:,i])
    x_train_zeros_scores[i] = cal_scores(x_train_zeros[:,i])
    x_test_ones_scores[i] = cal_scores(x_test_ones[:,i])
    x_test_zeros_scores[i] = cal_scores(x_test_zeros[:,i])


print(x_train_ones_scores)
print(x_train_zeros_scores)
print(x_test_ones_scores)
print(x_test_zeros_scores)



def gaussian_function(val,sigma,mean):
    scores = np.zeros(len(mean))
    for i in range(len(mean)):
        s1 = 1/(sigma[i] * math.sqrt(2*np.pi)) 
        s2 =  ((np.exp(-((val[i]-mean[i])**2))/(2*sigma[i]*sigma[i])))
        scores[i]=s1*s2

    return scores


train_pred = np.zeros(train_length)



for i in range(train_length):
    one_score = gaussian_function(x_train[i],x_train_ones_scores[:,1],x_train_ones_scores[:,0])
    zero_score = gaussian_function(x_train[i],x_train_zeros_scores[:,1],x_train_zeros_scores[:,0])
    one = np.prod(one_score,axis=0)
    zero = np.prod(zero_score,axis=0)
    if(one>=zero):
        train_pred[i]=1
    else:
        train_pred[i]=0


print(sum(train_pred==y_train)/train_length)


test_pred = np.zeros(test_length)


for i in range(test_length):
    one_score = gaussian_function(x_test[i],x_test_ones_scores[:,1],x_test_ones_scores[:,0])
    zero_score = gaussian_function(x_test[i],x_test_zeros_scores[:,1],x_test_zeros_scores[:,0])
    one = np.prod(one_score,axis=0)
    zero = np.prod(zero_score,axis=0)
    if(one>=zero):
        test_pred[i]=1
    else:
        test_pred[i]=0


print(sum(test_pred==y_test)/test_length)

