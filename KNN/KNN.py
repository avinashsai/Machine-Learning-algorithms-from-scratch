import os
import re
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('iris_train.txt',sep=',')

print(train_data.head(3))

train_data["label"] = (train_data["label"]=="Iris-versicolor").astype(int)

test_data = pd.read_csv('iris_test.txt',sep=',')

print(test_data.head(3))

test_data["label"] = (test_data["label"]=="Iris-versicolor").astype(int)

train_length = len(train_data)
print(train_length)

test_length = len(test_data)
print(test_length)

train_labels = np.zeros(train_length)

for i in range(train_length):
    train_labels[i]=train_data["label"][i]

test_labels = np.zeros(test_length)

for i in range(test_length):
    test_labels[i]=test_data["label"][i]

train_data = train_data.drop(["label"],axis=1)

x_train = train_data.as_matrix()

test_data = test_data.drop(["label"],axis=1)

x_test = test_data.as_matrix()

print(x_train.shape)

print(x_test.shape)

def minmaxscaler(col):
    col1 = (col-min(col))/(max(col))
    return col1

for column in train_data:
    train_data[column] = minmaxscaler(train_data[column])

for column in test_data:
    test_data[column] = minmaxscaler(test_data[column])

def euclidean_distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

def make_label(dis,k):
    count_0 = 0
    count_1 = 0
    for i in range(k):
        if(dis[i][1]==1):
            count_1+=1
        else:
            count_0+=1
    if(count_1>count_0):
        return 1
    else:
        return 0

def knn(k,train_data,test_data,train_labels,test_labels,train_length,test_length):
    pred = np.zeros(test_length)
    for i in range(test_length):
        p1 = test_data[i]
        dis = np.zeros((train_length,2))
        for j in range(train_length):
            p2 = train_data[j]
            dis[j][0] = euclidean_distance(p1,p2)
            dis[j][1] = train_labels[j]
        dis = sorted(dis,key=lambda val: val[0])
        pred[i] = make_label(dis,k)
    def_acc = sum(pred==test_labels)/test_length

    return def_acc

def sklearn_knn(k,train_data,test_data,train_labels,test_labels,train_length,test_length):
    
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_data,train_labels)
    pred = clf.predict(test_data)
    return accuracy_score(pred,test_labels)


def_accuracy = []
K_values = []
sk_accuracy = []


for k in range(3,20,2):
    count_def = 0
    count_sk = 0
    def_accu = knn(k,x_train,x_test,train_labels,test_labels,train_length,test_length)
    sklearn_accu = sklearn_knn(k,x_train,x_test,train_labels,test_labels,train_length,test_length)
    if(def_acc>sklearn_acc):
        count_def+=1
    elif(def_acc<sklearn_acc):
        count_sk+=1
    K_values.append(k)
    def_accuracy.append(def_accu)
    sk_accuracy.append(sklearn_accu)


print(K_values)


from matplotlib import pyplot as plt1
from matplotlib import pyplot as plt2


plt1.plot(K_values,def_accuracy,color='g')
plt1.show()


plt2.plot(K_values,sk_accuracy,color='r')
plt2.show()


if(count_def==count_sk):
    print("Draw!!")
elif(count_def>count_sk):
    print("You Win!!")
else:
    print("You Lost!!")

