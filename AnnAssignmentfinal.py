#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:11:39 2019

@author: vish
"""
import numpy as np
from sklearn import datasets

import math

def relu(X):
   return np.maximum(X, 0)


def feed_forward(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) ;

    #a1 = np.tanh(z1)
    
    a1 = relu(z1)
    print(a1);
    #print(z1,'\n',W2);
    z2 = a1.dot(W2) 
#    ztemp = np.asarray(z2)
    out = (z2);
    #print(out);
    #out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    #print(out);
    #print('hi',np.sum(out,axis=1,keepdims=True));
#    print('hello',z2)
#    print('hello there',out);
    return z1, a1, z2, out

def calculate_loss(model,X,Y):
    z1, a1, z2, out = feed_forward(model, X);
#    mse = (np.square(np.subtract(Y, out)));
    mse=Y-out;
    temp=[]
    for i in range(len(Y)):
        a = Y[i]-out[i];
        a = a*a;
        a = 0.5*a;
        temp.append(a);
        a=1
    print();
    #print("---------------",out);
    return mse;

def accuracy(loss,Y):
    num_right = 0.0;
    for i in range(len(Y)):
        if(loss[i] == Y[i]):
            num_right+=1.0;
    return num_right/float(len(loss))            
        

def backpropogation(Y,out,z2,a1,X,z1,model):
    #print(Y.shape,out.shape,z2.shape,a1.shape,X.shape,z1.shape)
    #temp=out.T.dot(1-out)
    dw2=((Y-out).T.dot(out)).dot((1-out).T.dot(a1));
    delta2 = out.dot(model['W2'].T) * (1 - np.power(a1, 2));
    dw1 = np.dot(X.T, delta2)
    print(dw1.shape,dw2.shape);
    dw2=dw2.T;
    return dw1,dw2;

def train(model,X,Y,number_epoch):
    for epoch in range(number_epoch):
        curr_loss = calculate_loss(model,X,Y);
        if accuracy(curr_loss,Y)==1.0 : break
#        print('current_epoch and current accuracy',epoch,accuracy(curr_loss,Y));
        z1, a1, z2, out = feed_forward(model, X);
        dw1,dw2 = backpropogation(Y,out,z2,a1,X,z1,model);
        model['W1'] += 0.5*dw1;
        model['W2'] += 0.5*dw2;    
    return model;

def main():
    iris = datasets.load_iris();
    X = iris.data;
    Y = iris.target;
    Y = np.reshape(Y, (Y.shape[0], 1))
    num_examples=len(X);
    #print(X,"------------\n", num_examples);
    #print(Y[0],Y[51],X[149]);
    print("---------------");
    W1 = np.random.rand(4,2);
    b1 = np.zeros((1,2));
    W2 = np.random.rand(2,1);
    b2 = np.zeros((1,1));
    #print (W1,"-------------------\ntere\n",W2);
    model = {}
    model['W1'] = W1;
    model['b1'] = b1;
    model['W2'] = W2;
    model['b2'] = b2;
    learning_rate=0.01;
    #print(calculate_loss(model,X,Y));
    model = train(model,X, Y, 5);
    print(' current accuracy',accuracy(calculate_loss(model,X,Y),Y))
    
    
if __name__ == '__main__':
    main()
