import os
import re
import numpy as np
import pandas as pd
import math

train_data = pd.read_csv('iris_train.txt',sep=',')

print(train_data.head(2))

train_length = len(train_data)

print(train_length)

train_data['label'] = np.where(train_data.label=="Iris-setosa",0,train_data.label)

train_data['label'] = np.where(train_data.label=="Iris-versicolor",1,train_data.label)


print(sum(train_data['label']==1))

print(sum(train_data['label']==0))


def scale_data(col):
    col1 = (col-min(col))/(max(col)-min(col))
    return col1

train_data["a"] = scale_data(train_data["a"])

train_data["b"] = scale_data(train_data["b"])

train_data["c"] = scale_data(train_data["c"])

train_data["d"] = scale_data(train_data["d"])

test_data = pd.read_csv('iris_test.txt',sep=',')

test_length = len(test_data)

test_data['label'] = np.where(test_data.label=="Iris-setosa",0,test_data.label)

test_data['label'] = np.where(test_data.label=="Iris-versicolor",1,test_data.label)


test_data["a"] = scale_data(test_data["a"])
test_data["b"] = scale_data(test_data["b"])
test_data["c"] = scale_data(test_data["c"])
test_data["d"] = scale_data(test_data["d"])


train_data = train_data.drop(["label"],axis=1)

test_data = test_data.drop(["label"],axis=1)

x_train = train_data.as_matrix()


x_test = test_data.as_matrix()


print(x_train.shape)

print(x_test.shape)

y_train = np.zeros((70,1))
y_train[35:70,]=1


y_test = np.zeros((30,1))
y_test[15:30,]=1


def measure_acc(test_pred,y_test):
    count = 0
    
    for i in range(len(test_pred)):
        if(y_test[i][0]==test_pred[i]):
            count = count+1
    print(count/30)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(a,size):
    labels = np.zeros(size)
    for i in range(size):
        if(a[0][i]>0.5):
            labels[i]=1
        else:
            labels[i]=0
    return labels


def cost_function(w,x,b,y_train,m,i):
    z = np.dot(w,x.T) + b
    a = sigmoid(z)
    
       
    p1 = np.dot(y_train.T ,(np.log(a)).T)
    p2 = np.dot((1-y_train).T ,(np.log(1-a)).T)
    
    cost = (-1/m) * sum(p1+p2)
    
    dw = (1/m) * (np.dot(x.T,(a-y_train.T).T))
    db = (1/m) * (np.sum(a-y_train.T))
    
    
    return dw,db,cost


def train(x_train,x_test,alpha,epochs,y_train):
    m = x_train.shape[0]
    n_features = x_train.shape[1]
    w = np.zeros((1,n_features))
    b = 0
    for i in range(epochs):
        dw,db,cost = cost_function(w,x_train,b,y_train,m,i)
        
        w = w - alpha * dw.T
        b = b - alpha * db
        
        
        if(i%1000==0):
            print(cost)
    return w,b
    

def logisticregression(x_train,x_test,y_train,y_test,alpha,epochs):
    
    w,b = train(x_train,x_test,alpha,epochs,y_train)
    z = np.dot(w,x_train.T) + b
    a = sigmoid(z)
    
    train_pred = predict(a,x_train.shape[0])
    
    z = np.dot(w,x_test.T) + b
    a = sigmoid(z)
    
    test_pred = predict(a,x_test.shape[0])
    
    measure_acc(test_pred,y_test)


alpha = 0.0001
epochs = 8000


logisticregression(x_train,x_test,y_train,y_test,alpha,epochs)

