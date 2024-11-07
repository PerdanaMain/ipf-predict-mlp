#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:42:03 2024

@author: aimo
"""

# =============================================================================
# from sklearn.neural_network import MLPRegressor
# import numpy as np
# import matplotlib.pyplot as pl
# 
# t = np.linspace(0,2,10000)
# data = np.sin(2*np.pi*8*t)
# fig, ax = pl.subplots()
# ax.plot(data)
# pl.show()
# 
# training = data[:len(data)//2]
# trainingplus1 = data[1:len(data)//2+1]
# test = data[len(data)//2:]
# reg = MLPRegressor()
# model = reg.fit(training.reshape(-1,1), trainingplus1)
# predtrain = np.array([])
# for i in training:
#     predtrain = np.append(predtrain, reg.predict(np.array([i]).reshape(-1, 1)))
# 
# fig, ax = pl.subplots()
# ax.plot(training,'b-')
# ax.plot(predtrain,'r--')
# ax.set_title("Train")
# pl.show()
# 
# predtest = np.array([])
# for i in test:
#     predtest = np.append(predtest, reg.predict(np.array([i]).reshape(-1, 1)))
# 
# fig, ax = pl.subplots()
# ax.plot(test,'b-')
# ax.plot(predtest,'r--')
# ax.set_title("Test")
# pl.show()
# 
# 
# predtest = np.array([])
# for idx, i in enumerate(test):
#     if len(predtest)==0:
#         predtest = np.append(predtest, reg.predict(np.array([i]).reshape(-1, 1)))
#     else:
#         predtest = np.append(predtest, reg.predict(predtest[idx-1].reshape(-1,1)))
# 
# fig, ax = pl.subplots()
# ax.plot(test,'b-')
# ax.plot(predtest,'r--')
# ax.set_title("Test recurrent")
# pl.show()
# 
# 
# # -------- data lain -----------
# 
# datanonnoise = np.arange(0,10000)
# t = np.linspace(0,2,10000)
# data = datanonnoise + np.sin(2*np.pi*8*t)*100
# fig, ax = pl.subplots()
# ax.plot(datanonnoise,'b-')
# ax.plot(data,'r--')
# ax.set_title("Data raw")
# pl.show()
# 
# train2 = data[:len(data)//2]
# trainplussatu = data[1:len(data)//2+1]
# test2 = data[len(data)//2:]
# 
# reg2 = MLPRegressor()
# model2 = reg2.fit(train2.reshape(-1,1), trainplussatu)
# predtest2 = np.array([])
# for idx, i in enumerate(test2):
#     if len(predtest2)==0:
#         predtest2 = np.append(predtest2, reg.predict(np.array([i]).reshape(-1, 1)))
#     else:
#         predtest2 = np.append(predtest2, reg2.predict(predtest2[idx-1].reshape(-1,1)))
# 
# fig, ax = pl.subplots()
# ax.plot(test2,'b-')
# ax.plot(predtest2,'r--')
# ax.set_title("Test recurrent 2")
# pl.show()
# =============================================================================


from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as pl

t = np.linspace(0,2,100)
data = np.sin(2*np.pi*8*t)
train = data[:len(data)//2]
trainplus1 = data[1:len(data)//2+1]
test = data[len(data)//2:]
fig, ax = pl.subplots()
ax.plot(data)
pl.show()

reg = MLPRegressor()
model = reg.fit(train.reshape(-1,1), trainplus1)
d = np.array([])
pred = np.array([])
for idx,i in enumerate(test):
    d = np.append(d, i)
    pred = model.predict(d.reshape(-1,1))
    fig, ax = pl.subplots()
    ax.plot(test,'b-')
    ax.plot(pred,'r--')
    fig.savefig(f"hasil/{idx}.jpg")
    pl.close(fig)
