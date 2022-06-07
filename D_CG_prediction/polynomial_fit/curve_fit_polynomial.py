import pylab
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import importlib
import numpy as np
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))


column=0
for i in range(2,8):
    row=int((i-2)/2)

    column=column+1
    if (i-2)%2==0:
        column=0
 
    f=open('new.py','w')
    s=''
    s1=''
    for j in range(0,i):
        s=s+'a'+str(j)
        s1=s1+'(a'+str(j)+'*x**'+str(j)+')'
        if j!=(i-1):
            s=s+','
            s1=s1+'+'
            
    s='def f(x,'+s+'):'
    s1='    y='+s1
    s2='    return y'
    f.write(s+'\n')
    f.write(s1+'\n')
    f.write(s2+'\n')
    f.close()
    os.system('cat new.py fit_new.py > tog.py')
    import tog
    importlib.reload(tog)

    np.savetxt(str(row)+'_'+str(column)+'y_train', tog.ydata_train)
    np.savetxt(str(row)+'_'+str(column)+'fitted_y_train', tog.fitted_y_train)
    np.savetxt(str(row)+'_'+str(column)+'y_test', tog.ydata_test)
    np.savetxt(str(row)+'_'+str(column)+'fitted_y_test', tog.fitted_y_test)



    axes[row,column].scatter(tog.ydata_train,tog.fitted_y_train, label=' Train: MAE = '+str(tog.mae_train)+', r$^2$ = '+str(tog.r2_train))
    axes[row,column].scatter(tog.ydata_test,tog.fitted_y_test, label='Test: MAE = '+str(tog.mae_test)+', r$^2$ = '+str(tog.r2_test))
    axes[row,column].plot(tog.ydata_train,tog.ydata_train, color='black')
    axes[row,column].legend(fontsize=15, loc=2)
    #axes[row,column].tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black', grid_alpha=10)
    axes[row,column].set_xlabel(r"Actual $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
    axes[row,column].set_ylabel(r"Fitted $D_{CG}$ $(in$ $10^{-9} m^{2} s^{-1})$",fontsize=15)
    axes[row,column].set_xlim(0,30)
    axes[row,column].set_ylim(0,30)
   # axes[row,column].text(-4,30,"(a)", size=15)
    #axes[row,column].set_title("abc", fontsize=25)     
    #os.system('python together.py')
''' 
axes[0,0].set_title(r"$f(x)=A_{0} + A_{1}x$", fontsize=15)
axes[0,1].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2}$", fontsize=15)
axes[0,2].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3}$", fontsize=15)
axes[1,0].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3} + A_{4}x^{4}$", fontsize=15)
axes[1,1].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3} + A_{4}x^{4}+ A_{5}x^{5}$", fontsize=15)
axes[1,2].set_title(r"$f(x)=A_{0} + A_{1}x + A_{2}x^{2} + A_{3}x^{3} + A_{4}x^{4}+ A_{5}x^{5} +A_{6}x^{6}$", fontsize=15)

axes[0,0].text(-4,30,"(a)", size=15)
axes[0,1].text(-4,30,"(b)", size=15)
axes[0,2].text(-4,30,"(c)", size=15)
axes[1,0].text(-4,30,"(d)", size=15)
axes[1,1].text(-4,30,"(e)", size=15)
axes[1,2].text(-4,30,"(f)", size=15)

#plt.xlabel('D_AA')
plt.tight_layout()
#plt.show()
'''






