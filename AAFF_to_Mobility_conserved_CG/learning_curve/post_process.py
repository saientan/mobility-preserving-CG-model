from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt



a=[]
b=[]
for i in range(2,26):
    j=i*10
    s='out_'+str(j)
    #f=open(s,'r')
    s1="grep 'Train on' "+s+" > number"
    os.system(s1)
    f=open('number','r')
    l_f=f.readlines()
    train_number=l_f[0].split()[2]
    #l_f=f.readlines()
    #l=len(l_f)
    s1="grep 'FULL_MAE' "+s+" > check"
    os.system(s1)
    f=open('check','r')
    l_f=f.readlines()
    l=len(l_f)
    r2_all=0
    for k in range(0,l):
        r2_all=r2_all+float(l_f[k].split()[1])
    f.close()
    a.append(j)
    b.append(r2_all/10)
    print (train_number,r2_all/10)

plt.plot(a,b)
#plt.xscale('log')

plt.show()

