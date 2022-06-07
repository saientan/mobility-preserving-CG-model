import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import sklearn
import numpy as np


train_real=[]
test_real=[]


train_RFR=[]
test_RFR=[]



def reg_stats(y_true,y_pred):
    r2 = sklearn.metrics.r2_score(y_true,y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return r2,mae





for i in range(0,10):
    data=pylab.loadtxt("y_real_train"+str(i)+".txt")
    for j in range(0,len(data)):
        train_real.append(data[j])

for i in range(0,10):
    data=pylab.loadtxt("y_real_test"+str(i)+".txt")
    for j in range(0,len(data)):
        test_real.append(data[j])
for i in range(0,10):
    data=pylab.loadtxt("y_RFR_train"+str(i)+".txt")
    for j in range(0,len(data)):
        train_RFR.append(data[j])
for i in range(0,10):
    data=pylab.loadtxt("y_RFR_test"+str(i)+".txt")
    for j in range(0,len(data)):
        test_RFR.append(data[j])


imp=np.zeros(19)
for i in range(0,10):
    data=pylab.loadtxt("feature_importance_"+str(i)+".txt")
    for j in range(0,len(data)):
        imp[j]=imp[j]+(data[j])

for i in range(0,19):
    imp[i]=imp[i]/10







r2_train, mae_train= reg_stats(train_real,train_RFR)
r2_train=round(r2_train,3)
mae_train=round(mae_train,3)
#print ("Train", r2, mae)

r2_test, mae_test= reg_stats(test_real,test_RFR)
r2_test=round(r2_test,3)
mae_test=round(mae_test,3)
#print ("Train", r2, mae)
#print ("Test", r2, mae)



plt.scatter(train_real,train_RFR,label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
plt.scatter(test_real,test_RFR,label=' Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
plt.xlabel(r"Actual $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
plt.ylabel(r"Predicted $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
plt.plot(train_real,train_real, color='black')
plt.legend(fontsize=15, loc=2)
plt.show()

feature_names=[r"$K^{CC}$",r"$r_{0}^{CC}$",r"$K^{CH}$",r"$r_{0}^{CH}$",r"$K^{CCC}$",r"$\theta_{0}^{CCC}$",r"$K^{CCH}$",r"$\theta_{0}^{CCH}$",r"$K^{HCH}$",r"$\theta_{0}^{HCH}$", r"$K1^{CCCC}$",r"$K2^{CCCC}$",r"$K3^{CCCC}$", r"$K1^{CCCH}$",r"$K2^{CCCH}$",r"$K3^{CCCH}$", r"$K1^{HCCH}$",r"$K2^{HCCH}$",r"$K3^{HCCH}$"]


#feature_names=[r"$K^{B1}$",r"$r_{0}^{B1}$",r"K_B2",r"$r_{0}$_B2$",r"K_A1$)",r"$\theta_{0}$",r"K_A2",r"$\theta_{0}$_A2$",r"K_A3($kcalmol$)",r"$\theta_{0}$", r"K1_D1",r"K2_D1",r"K3_D1()", r"K1_D2()",r"K2_D2()",r"K3_D2()", r"K1_D3()",r"K2_D3()",r"K3_D3()"]

plt.barh(feature_names, imp, color='red')
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
plt.xlabel(r"Feature Importance Rank",fontsize=15)
plt.ylabel(r"Feature Names",fontsize=15)
plt.show()




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))

axes[0].scatter(train_real,train_RFR,label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
axes[0].scatter(test_real,test_RFR,label=' Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
axes[0].plot(train_real,train_real, color='black')
axes[0].set_xlabel(r"Actual $D_{AA}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
axes[0].set_ylabel(r"Predicted $D_{AA}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
axes[0].plot(train_real,train_real, color='black')
axes[0].legend(fontsize=15, loc=2)
axes[0].tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
axes[0].set_xlim(-0.65,10)
axes[0].set_ylim(-0.65,10)


axes[1].barh(feature_names, imp, color='red')
#axes[0][1].se_tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
axes[1].set_xlabel(r"Feature Importance Rank",fontsize=15)
axes[1].set_ylabel(r"Feature Names",fontsize=15)
axes[1].tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
#plt.show()


fig.tight_layout()
plt.savefig('Figure_SI_2.png', dpi=200)
plt.show()



































