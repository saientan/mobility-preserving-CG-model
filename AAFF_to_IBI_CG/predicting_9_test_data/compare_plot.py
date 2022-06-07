from __future__ import division
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
import matplotlib.pyplot as plt
actual=np.zeros((9,900))
predic=np.zeros((9,900))


for i in range(1,11):
    data_actual=pylab.loadtxt('actual_'+str(i))
    data_predic=pylab.loadtxt('predic_'+str(i))
    
    for j in range(0,len(data_actual)):
        for k in range(0,len(data_actual[j])):
            actual[j][k]=actual[j][k]+data_actual[j][k]
            predic[j][k]=predic[j][k]+data_predic[j][k]


for j in range(0,len(data_actual)):
    for k in range(0,len(data_actual[j])):
        actual[j][k]=actual[j][k]/10
        predic[j][k]=predic[j][k]/10



fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

k=0
a=[]
b=[]
for i in range(0,9):
    #r2_train = round(r2_score(ytrain[:,i], y_pred_train[:,i]),3)
    #r2_test = round(r2_score(ytest[i], y_pred_test[i]),3)
    #print (i, "r2_test =", r2_test)
    j=int(i/3)
    k=k+1
    if i%3==0:
        k=0
        #r2_train = round(r2_score(ytrain[:,i], y_pred_train[:,i]),3)
    r2_test = round(r2_score(actual[i], predic[i]),3)
    mae_test = round(mean_absolute_error(actual[i], predic[i]),3)
    print (r2_test, mae_test)
        #print (i, "r2_train =", r2_train, "r2_test =", r2_test)
       # MAE_train= round(mean_absolute_error(y_train_unscaled[:,i],y_pred_train_unscaled[:,i]),2)
        #MAE_test= round(mean_absolute_error(y_test_unscaled[:,i],y_pred_test_unscaled[:,i]),2)
        #print (axes_xlabel[i], "r2_test =",r2_test, "MAE_test =", MAE_test)
        #axes[j,k].set_xlabel(axes_xlabel[i])
        #axes[j,k].set_ylabel(axes_xlabel[i])
        #axes[j,k].plot(y_train_unscaled[:,i], y_train_unscaled[:,i], color='black')
        #axes[j,k].scatter(y_train_unscaled[:,i], y_pred_train_unscaled[:,i], label="Train :"+"\n" + "MAE= "+str(MAE_train)+", r2= "+str(r2_train),color='red')
    #a=[]
    #b=[]
    #c=[]
    #for l in range(0,len(actual[i])):
    #    a.append(actual[i][l])
    #    b.append(predic[i][l])
    #    c.append(l)
    #print (j,k,a,b,c) 
  
    dist=pylab.loadtxt('nb.A-A.pot.table')
    dist=dist[:,1]   
    b=[]
    for index in range(500,500+900):
        b.append(dist[index])
   

    axes[j,k].set_ylabel(r"$U^{CG}(r)$ $(kcal$ $mol^{-1})$",fontsize=18)
    axes[j,k].set_xlabel(r"$r$ $(\AA)$",fontsize=18)
    


    axes[j,k].legend(fontsize=18, loc=2)
    axes[j,k].tick_params(direction='in', length=6, width=2, colors='black', labelsize=18, grid_color='black')

    axes[j,k].plot(b,actual[i], label="IBI derived CG potential")
    axes[j,k].plot(b,predic[i],label="Predicted CG potential")
    axes[j,k].legend(fontsize=18, loc=1)
    axes[j,k].set_title(r'$MAE =$'+str(mae_test)+' $kcal$ $mol^{-1}$', fontsize=18)

    axes[j,k].set_ylim(-2,2)

    


axes[0,0].text(4,2.4,"(a)", size=18)
axes[0,1].text(4,2.4,"(b)", size=18)
axes[0,2].text(4,2.4,"(c)", size=18)
axes[1,0].text(4,2.4,"(d)", size=18)
axes[1,1].text(4,2.4,"(e)", size=18)
axes[1,2].text(4,2.4,"(f)", size=18)
axes[2,0].text(4,2.4,"(g)", size=18)
axes[2,1].text(4,2.4,"(h)", size=18)
axes[2,2].text(4,2.4,"(i)", size=18)


fig.tight_layout()

plt.savefig('Figure_8.png', dpi=200)


plt.show()


print (r2_test)   

