def f(x,a0,a1,a2,a3,a4,a5,a6):
    y=(a0*x**0)+(a1*x**1)+(a2*x**2)+(a3*x**3)+(a4*x**4)+(a5*x**5)+(a6*x**6)
    return y
import pylab
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#def f(x,a0,a1,a2,a3,a4,a5,a6,a7):
#    y=(a0*x**0)+(a1*x**1)+(a2*x**2)+(a3*x**3)+(a4*x**4)+(a5*x**5)+(a6*x**6)+(a7*x**7)
#    return y
    
def fitt(training_data):
    data=pylab.loadtxt(training_data)
    xdata=data[:,0]
    ydata=data[:,1]
#plt.scatter(xdata,ydata)
    param1, param2=curve_fit(f, xdata, ydata)
    return param1,param2

def plot_data(data_name,param1):
    data=pylab.loadtxt(data_name)
    xdata=data[:,0]
    ydata=data[:,1]
    alpha=data[:,2]
    fitted_y=np.zeros(len(ydata))
    fitted_alpha=np.zeros(len(ydata))
    for i in range(0,len(param1)):
        fitted_y=(param1[i]*xdata**i)+fitted_y
    for i in range(0,len(fitted_y)):
        fitted_alpha[i]=fitted_y[i]/xdata[i]
    mae,r2=r2_mae(alpha,fitted_alpha)
    #figure=plt.scatter(ydata,fitted_y, label='mae='+str(mae)+'r2='+str(r2))
    #plt.plot(ydata,ydata)
    return alpha,fitted_alpha,mae,r2

def r2_mae(y_true,y_pred):
    mae=round((mean_absolute_error(y_true, y_pred)),3)
    r2=round((r2_score(y_true, y_pred)),3)
    return mae, r2
param1,param2=fitt('training_data.dat')
ydata_train,fitted_y_train,mae_train,r2_train=plot_data('training_data.dat',param1)
ydata_test,fitted_y_test,mae_test,r2_test=plot_data('testing_data.dat',param1)
#plt.legend()
#plt.show()








