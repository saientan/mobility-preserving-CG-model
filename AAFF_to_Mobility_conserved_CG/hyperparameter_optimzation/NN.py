import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pylab
import datetime
import os
from models import getmodel,model_training
from sklearn.preprocessing import StandardScaler
import sklearn
import random
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
from sklearn.model_selection import KFold
# Some System Info
print("Tested for tf-gpu=2.1.0. This tf version: ",tf.__version__)
#print("List of GPUs found: ", tf.config.experimental.list_physical_devices('GPU'))

# Model Labeling
model_name = "Molecule"
label_name = "Energy"

#Data for training


from numpy.random import seed
seed(1)

tf.random.set_seed(1)


#np.random.seed(1)
#random.seed(1)
#data=pylab.loadtxt('training_data.dat')
data=pylab.loadtxt('shuffled_all_data_r2_alpha_g_r_restricted.dat')
x=data[:,3:22]
#x=data[:,0]
#print (x)
#x=x.reshape(-1, 1)
#y=data[:,20]
#sid=random.uniform(1,100000)
#y=data[:,2154:2994]
y=data[:,1484:2384]
kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(x)
sid=random.uniform(1,100000)
for train_index, test_index in kf.split(x):

#y=data[:,19]
#print(y)
#x=data[:,0]
#y=data[:,1]

#y=y.reshape(-1,1)
    #y_scaler = StandardScaler()
#y = y_scaler.fit_transform(y.reshape(-1,1))
    #y = y_scaler.fit_transform(y)
    x_scaler = StandardScaler()
    x = x_scaler.fit_transform(x)
    xtrain, xtest = x[train_index], x[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    #xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x, y, test_size=0.3, random_state=1)

#for i in range(0,len(xtrain)):
#    print (xtrain[i])
#print (xtrain, xtest)

    print(xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)

#np.expand_dims(y,axis=-1) # If not shape (batch,1)

#Manual Train/Validation Split

#Training  
    model = getmodel(xtrain.shape[-1], sid)
#os.makedirs("fit", exist_ok=True)
#os.makedirs("fit")
    y_pred_train, y_pred_test= model_training(model_name,model,xtrain,ytrain,xtest,ytest,sid)



    #y_test_unscaled = y_scaler.inverse_transform(ytest)
    #y_train_unscaled = y_scaler.inverse_transform(ytrain)
    #y_pred_test_unscaled = y_scaler.inverse_transform(y_pred_test)
    #y_pred_train_unscaled = y_scaler.inverse_transform(y_pred_train)

    x_train_unscaled = x_scaler.inverse_transform(xtrain)

    a =[]
    b =[]
    for i in range(0,len(ytest)):
        for j in range(0,len(ytest[i])):
            a.append(ytest[i][j])
            b.append(y_pred_test[i][j])

    print ("FULL_test", r2_score(a,b))


#for i in range(0,len(x_train_unscaled)):
#    print (x_train_unscaled[i])

    #fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))


    #axes_xlabel=[r"K_B1($kcalmol^{-1}\AA^{-2}$)",r"$r_{0}$_B1$(\AA)$",r"K_B2($kcalmol^{-1}\AA^{-2}$)",r"$r_{0}$_B2$(\AA)$",r"K_A1($kcalmol^{-1}rad^{-2}$)",r"$\theta_{0}$_A1$(deg)$",r"K_A1($kcalmol^{-1}rad^{-2}$)",r"$\theta_{0}$_A2$(deg)$",r"K_A3($kcalmol^{-1}rad^{-2}$)",r"$\theta_{0}$_A3$(deg)$", r"K1_D1($kcalmol^{-1}$)",r"K2_D1($kcalmol^{-1}$)",r"K3_D1($kcalmol^{-1}$)", r"K1_D2($kcalmol^{-1}$)",r"K2_D2($kcalmol^{-1}$)",r"K3_D2($kcalmol^{-1}$)", r"K1_D3($kcalmol^{-1}$)",r"K2_D3($kcalmol^{-1}$)",r"K3_D3($kcalmol^{-1}$)"]


    #k=0
    #for i in range(0,19):
    #    j=int(i/4)

    #    k=k+1
    #    if i%4==0:
    #        k=0
    #    r2_train = round(r2_score(ytrain[:,i], y_pred_train[:,i]),3) 
    #    r2_test = round(r2_score(ytest[:,i], y_pred_test[:,i]),3)
    #    MAE_train= round(mean_absolute_error(y_train_unscaled[:,i],y_pred_train_unscaled[:,i]),2)
    #    MAE_test= round(mean_absolute_error(y_test_unscaled[:,i],y_pred_test_unscaled[:,i]),2)
    #    print (axes_xlabel[i], "r2_test =",r2_test, "MAE_test =", MAE_test)
    #    print (axes_xlabel[i], "r2_train =",r2_train, "MAE_train =", MAE_train)
    #    axes[j,k].set_xlabel(axes_xlabel[i])
    #    axes[j,k].set_ylabel(axes_xlabel[i])
    #    axes[j,k].plot(y_train_unscaled[:,i], y_train_unscaled[:,i], color='black')
    #    axes[j,k].scatter(y_train_unscaled[:,i], y_pred_train_unscaled[:,i], label="Train :"+"\n" + "MAE= "+str(MAE_train)+", r2= "+str(r2_train),color='red')
    #    axes[j,k].scatter(y_test_unscaled[:,i], y_pred_test_unscaled[:,i], label="Test :"+"\n" + "MAE= "+str(MAE_test)+", r2= "+str(r2_test),color='blue')
    #    axes[j,k].legend()
#plt.plot(y_train_unscaled, y_train_unscaled, color='black')
#plt.scatter(y_pred_train_unscaled, y_train_unscaled, label="Training",color='blue')
#plt.scatter(y_pred_test_unscaled, y_test_unscaled, label="Test:",color='red')
    #fig.tight_layout()

#plt.show()











