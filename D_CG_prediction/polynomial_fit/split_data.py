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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
def r2_mae(y_true,y_pred):
    mae=round((mean_absolute_error(y_true, y_pred)),3)
    r2=round((r2_score(y_true, y_pred)),3)
    return mae, r2


from numpy.random import seed
seed(1)

tf.random.set_seed(1)



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
    data_train, data_test=data[train_index], data[test_index]
    np.savetxt('training_data.dat',data_train)
    np.savetxt('testing_data.dat',data_test)
    os.system('python curve_fit_polynomial.py')
    column=0
    for i in range(2,8):
        row=int((i-2)/2)

        column=column+1
        if (i-2)%2==0:
            column=0
        s1='mv '+str(row)+'_'+str(column)+'y_train '+str(row)+'_'+str(column)+'y_train_'+str(count)
        os.system(s1)
        s1='mv '+str(row)+'_'+str(column)+'fitted_y_train '+str(row)+'_'+str(column)+'fitted_y_train_'+str(count)
        os.system(s1)
        s1='mv '+str(row)+'_'+str(column)+'y_test '+str(row)+'_'+str(column)+'y_test_'+str(count)
        os.system(s1)
        s1='mv '+str(row)+'_'+str(column)+'fitted_y_test '+str(row)+'_'+str(column)+'fitted_y_test_'+str(count)
        os.system(s1)


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))


column=0
for i in range(2,8):
    row=int((i-2)/2)

    column=column+1
    if (i-2)%2==0:
        column=0
    
    train=[]
    pred_train=[]
    test=[]
    pred_test=[]
    for count in range(0,9):
        data=pylab.loadtxt(str(row)+'_'+str(column)+'y_train_'+str(count))
        for new in range(0,len(data)):
            train.append(data[new])
        data=pylab.loadtxt(str(row)+'_'+str(column)+'fitted_y_train_'+str(count))
        for new in range(0,len(data)):
            pred_train.append(data[new])
        data=pylab.loadtxt(str(row)+'_'+str(column)+'y_test_'+str(count))
        for new in range(0,len(data)):
            test.append(data[new])
        data=pylab.loadtxt(str(row)+'_'+str(column)+'fitted_y_test_'+str(count))
        for new in range(0,len(data)):
            pred_test.append(data[new])
    mae_train,r2_train=r2_mae(train,pred_train)
    mae_test,r2_test=r2_mae(test,pred_test)
    axes[row,column].scatter(train,pred_train, label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
    axes[row,column].scatter(test,pred_test, label='Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
    axes[row,column].plot(train,train, color='black')
    axes[row,column].legend(fontsize=20, loc=2)
    axes[row,column].tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')
    axes[row,column].set_xlabel(r"Actual $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=20)
    axes[row,column].set_ylabel(r"Fitted $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=20)
    axes[row,column].set_xlim(0,30)
    axes[row,column].set_ylim(0,30)
   # axes[row,column].text(-4,30,"(a)", size=15)
    #axes[row,column].set_title("abc", fontsize=25)
    #os.system('python together.py')
axes[0,0].set_title(r"$f(x)=A_{0} + A_{1}x$", fontsize=20)
axes[0,1].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2}$", fontsize=20)
axes[1,0].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3}$", fontsize=20)
axes[1,1].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3} + A_{4}x^{4}$", fontsize=20)
axes[2,0].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3} + A_{4}x^{4}+ A_{5}x^{5}$", fontsize=20)
axes[2,1].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3} + A_{4}x^{4}+ A_{5}x^{5} +A_{6}x^{6}$", fontsize=20)

axes[0,0].text(-4,30,"(a)", size=20)
axes[0,1].text(-4,30,"(b)", size=20)
axes[1,0].text(-4,30,"(c)", size=20)
axes[1,1].text(-4,30,"(d)", size=20)
axes[2,0].text(-4,30,"(e)", size=20)
axes[2,1].text(-4,30,"(f)", size=20)

#plt.xlabel('D_AA')
plt.tight_layout()
plt.savefig('Figure_1.png',dpi=200)
plt.show()








