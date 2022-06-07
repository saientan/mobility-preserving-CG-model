import os
import sys
import numpy as np
from scipy import stats
import math
#import torch
from matplotlib import pyplot as plt
import matplotlib
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import sklearn.linear_model
import tensorflow as tf
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
import pylab
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from sklearn.model_selection import KFold



def reg_stats(y_true,y_pred):
    r2 = sklearn.metrics.r2_score(y_true,y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return r2,mae

data=pylab.loadtxt('edited_extended_combined_training_testing_data.dat')
x1=data[:,0]
x1=x1.reshape(-1, 1)
x2=data[:,3:22]
x=np.hstack([x1,x2])
y=data[:,1]
y=y.reshape(-1, 1)


kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(x)
count=-1
for train_index, test_index in kf.split(x):
    count=count+1
    xtrain, xtest = x[train_index], x[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    
    print(xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)

    model =  RandomForestRegressor(max_depth=4)
    #model =  GradientBoostingRegressor()
    #model = linear_model.LinearRegression()
    model.fit(xtrain,ytrain.ravel())

    

    feature_names=[r"D_AA",r"K_B1($kcalmol^{-1}\AA^{-2}$)",r"$r_{0}$_B1$(\AA)$",r"K_B2($kcalmol^{-1}\AA^{-2}$)",r"$r_{0}$_B2$(\AA)$",r"K_A1($kcalmol^{-1}rad^{-2}$)",r"$\theta_{0}$_A1$(deg)$",r"K_A2($kcalmol^{-1}rad^{-2}$)",r"$\theta_{0}$_A2$(deg)$",r"K_A3($kcalmol^{-1}rad^{-2}$)",r"$\theta_{0}$_A3$(deg)$", r"K1_D1($kcalmol^{-1}$)",r"K2_D1($kcalmol^{-1}$)",r"K3_D1($kcalmol^{-1}$)", r"K1_D2($kcalmol^{-1}$)",r"K2_D2($kcalmol^{-1}$)",r"K3_D2($kcalmol^{-1}$)", r"K1_D3($kcalmol^{-1}$)",r"K2_D3($kcalmol^{-1}$)",r"K3_D3($kcalmol^{-1}$)"]

    plt.barh(feature_names, model.feature_importances_)
    plt.show()

    np.savetxt("feature_importance_"+str(count)+".txt", model.feature_importances_)
    y_pred_train = model.predict(xtrain)
    y_pred_test = model.predict(xtest)
    np.savetxt("y_real_test"+str(count)+".txt", ytest)
    np.savetxt("y_real_train"+str(count)+".txt", ytrain)
    np.savetxt("y_RFR_test"+str(count)+".txt", y_pred_test)
    np.savetxt("y_RFR_train"+str(count)+".txt", y_pred_train)


    r2_train, mae_train= reg_stats(ytrain,y_pred_train)
    r2_train=round(r2_train,3)
    mae_train=round(mae_train,3)
#print ("Train", r2, mae)

    r2_test, mae_test= reg_stats(ytest,y_pred_test)
    r2_test=round(r2_test,3)
    mae_test=round(mae_test,3)
#print ("Train", r2, mae)
#print ("Test", r2, mae)



    plt.scatter(ytrain,y_pred_train,label=' Train: MAE = '+str(mae_train)+', r$^2$ = '+str(r2_train))
    plt.scatter(ytest,y_pred_test,label=' Test: MAE = '+str(mae_test)+', r$^2$ = '+str(r2_test))
    plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
    plt.xlabel(r"Actual $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
    plt.ylabel(r"Predicted $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
    plt.plot(ytrain,ytrain, color='black')
    plt.legend(fontsize=15, loc=2)
    plt.show()



