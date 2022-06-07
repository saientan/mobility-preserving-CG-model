from __future__ import division
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import sklearn
import numpy as np



def reg_stats(y_true,y_pred):
    r2 = sklearn.metrics.r2_score(y_true,y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    return r2,mae


def plots(location, feature_names):
    train_real=[]
    test_real=[]
    train_RFR=[]
    test_RFR=[]

    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_real_train"+str(i)+".txt")
        for j in range(0,len(data)):
            train_real.append(data[j])

    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_real_test"+str(i)+".txt")
        for j in range(0,len(data)):
            test_real.append(data[j])
    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_RFR_train"+str(i)+".txt")
        for j in range(0,len(data)):
            train_RFR.append(data[j])
    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_RFR_test"+str(i)+".txt")
        for j in range(0,len(data)):
            test_RFR.append(data[j])


    imp=np.zeros(20)
    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"feature_importance_"+str(i)+".txt")
        for j in range(0,len(data)):
            imp[j]=imp[j]+(data[j])

    for i in range(0,20):
        imp[i]=imp[i]/10

    print (imp)

    r2_train, mae_train= reg_stats(train_real,train_RFR)
    r2_train=round(r2_train,3)
    mae_train=round(mae_train,3)


    r2_test, mae_test= reg_stats(test_real,test_RFR)
    r2_test=round(r2_test,3)
    mae_test=round(mae_test,3)


    return train_real,train_RFR,test_real,test_RFR, r2_train, mae_train, r2_test,mae_test, feature_names,imp




def plot_new(location):
    train_real=[]
    test_real=[]
    train_RFR=[]
    test_RFR=[]

    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_real_train"+str(i)+".txt")
        for j in range(0,len(data)):
            train_real.append(data[j])

    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_real_test"+str(i)+".txt")
        for j in range(0,len(data)):
            test_real.append(data[j])
    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_RFR_train"+str(i)+".txt")
        for j in range(0,len(data)):
            train_RFR.append(data[j])
    for i in range(0,10):
        data=pylab.loadtxt(str(location)+"y_RFR_test"+str(i)+".txt")
        for j in range(0,len(data)):
            test_RFR.append(data[j])
    r2_train, mae_train= reg_stats(train_real,train_RFR)
    r2_train=round(r2_train,3)
    mae_train=round(mae_train,3)


    r2_test, mae_test= reg_stats(test_real,test_RFR)
    r2_test=round(r2_test,3)
    mae_test=round(mae_test,3)


    return train_real,train_RFR,test_real,test_RFR, r2_train, mae_train, r2_test,mae_test




feature_names=[r"$D_{AA}$",r"$K^{CC}$",r"$r_{0}^{CC}$",r"$K^{CH}$",r"$r_{0}^{CH}$",r"$K^{CCC}$",r"$\theta_{0}^{CCC}$",r"$K^{CCH}$",r"$\theta_{0}^{CCH}$",r"$K^{HCH}$",r"$\theta_{0}^{HCH}$", r"$K1^{CCCC}$",r"$K2^{CCCC}$",r"$K3^{CCCC}$", r"$K1^{CCCH}$",r"$K2^{CCCH}$",r"$K3^{CCCH}$", r"$K1^{HCCH}$",r"$K2^{HCCH}$",r"$K3^{HCCH}$"]


train_real,train_RFR,test_real,test_RFR, r2_train, mae_train, r2_test,mae_test, feature_names,imp=plots("../ML_D_and_all_FF/", feature_names)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 28))

fig.tight_layout()
axes[1,0].set_xlabel(r"Actual $\alpha$",fontsize=20)
axes[1,0].set_ylabel(r"Predicted $\alpha$",fontsize=20)
axes[1,0].scatter(train_real,train_RFR,label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
axes[1,0].scatter(test_real,test_RFR,label=' Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
axes[1,0].plot(train_real,train_real, color='black')
axes[1,0].tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')
axes[1,0].legend(fontsize=20, loc=2)
axes[1,0].set_title(r"$D_{AA}$ + All FF parameters", fontsize=20)


plt.tight_layout()

axes[1,1].barh(feature_names, imp)
#axes[0][1].se_tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')
axes[1,1].set_xlabel(r"Feature Importance Rank",fontsize=20)
axes[1,1].set_ylabel(r"Feature Names",fontsize=20)
axes[1,1].tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')
#axes[j,k].legend()
axes[1,1].set_title(r"$D_{AA}$ + All FF parameters", fontsize=20)


feature_names=[r"$D_{AA}$",r"$K^{CC}$",r"$r_{0}^{CC}$",r"K_CH",r"$r_{0}_CH$",r"K_A1$)",r"$\theta_{0}$",r"K_A2",r"$\theta_{0}$_A2$",r"K_A3($kcalmol$)",r"$\theta_{0}$", r"K1_D1",r"K2_D1",r"K3_D1()", r"K1_D2()",r"K2_D2()",r"K3_D2()", r"K1_D3()",r"K2_D3()",r"K3_D3()"]


train_real,train_RFR,test_real,test_RFR, r2_train, mae_train, r2_test,mae_test=plot_new("../ML_without_D/")


fig.tight_layout()
axes[0,1].set_xlabel(r"Actual $\alpha$",fontsize=20)
axes[0,1].set_ylabel(r"Predicted $\alpha$",fontsize=20)
axes[0,1].scatter(train_real,train_RFR,label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
axes[0,1].scatter(test_real,test_RFR,label=' Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
axes[0,1].plot(train_real,train_real, color='black')
axes[0,1].legend(fontsize=20, loc=2)
axes[0,1].tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')

axes[0,1].set_title(r"All FF parameters", fontsize=20)

feature_names=[r"$D_{AA}$"]


train_real,train_RFR,test_real,test_RFR, r2_train, mae_train, r2_test,mae_test, =plot_new("../ML_with_only_D/")



axes[0,0].set_xlabel(r"Actual $\alpha$",fontsize=20)
axes[0,0].set_ylabel(r"Predicted $\alpha$",fontsize=20)
axes[0,0].scatter(train_real,train_RFR,label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
axes[0,0].scatter(test_real,test_RFR,label=' Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
axes[0,0].plot(train_real,train_real, color='black')
axes[0,0].legend(fontsize=20, loc=2)
axes[0,0].set_title(r"$D_{AA}$", fontsize=20)
axes[0,0].tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')
feature_names=[r"$D_{AA}$",r"$r_{0}^{B1}$",r"$r_{0}$_B2$",r"$\theta_{0}$"]


train_real,train_RFR,test_real,test_RFR, r2_train, mae_train, r2_test,mae_test=plot_new("../ML_with_important_features/4_input/")



axes[2,0].set_xlabel(r"Actual $\alpha$",fontsize=20)
axes[2,0].set_ylabel(r"Predicted $\alpha$",fontsize=20)
axes[2,0].scatter(train_real,train_RFR,label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
axes[2,0].scatter(test_real,test_RFR,label=' Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
axes[2,0].plot(train_real,train_real, color='black')
axes[2,0].legend(fontsize=20, loc=2)
axes[2,0].tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')
axes[2,0].set_title(r"4 most imporant features", fontsize=20)

feature_names=[r"$D_{AA}$",r"$r_{0}^{B2}$"]


train_real,train_RFR,test_real,test_RFR, r2_train, mae_train, r2_test,mae_test=plot_new("../ML_with_important_features/2_input/")



axes[2,1].set_xlabel(r"Actual $\alpha$",fontsize=20)
axes[2,1].set_ylabel(r"Predicted $\alpha$",fontsize=20)
axes[2,1].scatter(train_real,train_RFR,label=' Train: MAE = '+str(mae_train)+', R$^2$ = '+str(r2_train))
axes[2,1].scatter(test_real,test_RFR,label=' Test: MAE = '+str(mae_test)+', R$^2$ = '+str(r2_test))
axes[2,1].plot(train_real,train_real, color='black')
axes[2,1].legend(fontsize=20, loc=2)
axes[2,1].tick_params(direction='in', length=6, width=2, colors='black', labelsize=20, grid_color='black')
axes[2,1].set_title(r"2 most imporant features", fontsize=20)


axes[0,0].text(-4,32,"(a)", size=20)
axes[0,1].text(-4,32,"(b)", size=20)
axes[1,0].text(-4,32,"(c)", size=20)
axes[1,1].text(-0.05,21,"(d)", size=20)
axes[2,0].text(-4,32,"(e)", size=20)
axes[2,1].text(-4,32,"(f)", size=20)

fig.tight_layout()

plt.savefig('Figure_3.png', dpi=200)
plt.show()



