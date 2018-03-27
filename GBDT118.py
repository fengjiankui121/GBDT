import scipy.io as sio
import pywt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from scipy.signal import butter,lfilter
dataset=sio.loadmat(r'E:\python\GBDT\data_set_IVa_ay.mat')
y=sio.loadmat(r'E:\python\GBDT\true_labels_ay.mat')['true_y'][0]
sample=dataset['cnt']*0.1
pos=dataset['mrk']['pos'][0][0][0]
data=np.zeros((280,300,118))
for i in range(280):
    data[i]=sample[pos[i]:pos[i]+300]
b,a=butter(5,[8/50,30/50],btype='bandpass')
samplefilted=np.zeros((280,118,300))
for i in range(280):
    for j in range(118):
        samplefilted[i,j]=lfilter(b,a,data[i,:,j])
class1=samplefilted[y==1]
class2=samplefilted[y==2]
a=range(140)
b1=np.reshape(a,(10,14))
acc=np.zeros(10)
for k in range(10):
    b2=np.setdiff1d(a,b1[k])
    class1_train=class1[b2]
    class2_train=class2[b2]
    class1_test=class1[b1[k]]
    class2_test=class2[b1[k]]
    R1=np.zeros((118,118))
    R2=np.zeros((118,118))
    for i in range(126):
        R1=R1+np.cov(class1[i])
        R2=R2+np.cov(class2[i])
    R1=R1/126
    R2=R2/126
    R3=R1+R2
    Sigma, U0 = np.linalg.eig(R3)
    P = np.dot(np.diag(Sigma ** (-0.5)), U0.T)
    YL = np.dot(np.dot(P, R1), P.T)
    SigmaL, UL = np.linalg.eig(YL)
    I = np.argsort(SigmaL)
    F1 = np.dot(UL.T, P)[[0,1,116,117], :]
    featuref2 = np.zeros((126, 4))
    featuref3 = np.zeros((126, 4))
    for j in range(126):
        dataf1 = np.dot(class1_train[j, :, :].T, F1.T)
        dataf2 = np.dot(class2_train[j, :, :].T, F1.T)
        for jj in range(4):
            featuref2[j, jj] = np.log(np.var(dataf1[:, jj]))
            featuref3[j, jj] = np.log(np.var(dataf2[:, jj]))
    traindata = np.vstack((featuref2, featuref3))
    tes1 = np.zeros((14, 4))
    tes2 = np.zeros((14, 4))
    for j in range(14):
        dataf1 = np.dot(class1_test[j, :, :].T, F1.T)
        dataf2 = np.dot(class2_test[j, :, :].T, F1.T)
        for jj in range(4):
            tes1[j, jj] = np.log(np.var(dataf1[:, jj]))
            tes2[j, jj] = np.log(np.var(dataf2[:, jj]))
    testdata = np.vstack((tes1, tes2))
    trainlabel = np.hstack((np.ones((126)), np.ones((126)) * 2))
    testlabel = np.hstack((np.ones((14)), np.ones((14)) * 2))
    gbdt = GradientBoostingClassifier(
        loss='deviance'
        , learning_rate=0.1     #学习率
        , n_estimators=100       #弱学习器的个数
        , max_depth=3
        , init=None
        , random_state=None
        , max_features=None
        , max_leaf_nodes=None
        , warm_start=False
    )
    gbdt.fit(traindata,trainlabel)
    predict_testlabel=gbdt.predict(testdata)
    acc[k] = sum(predict_testlabel == np.squeeze(testlabel)) / 28

print(np.mean(acc))

