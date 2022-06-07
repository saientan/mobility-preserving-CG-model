import os
for i in range(1,26):
    j=i*10
    s1='shuffled_all_data_r2_alpha_g_r_restricted.dat'
    s="head -"+str(j)+" "+s1+" > learning_data.dat"
    print (s)
    os.system(s)
    s='python NN_with_optimized_parameter_KFOLD.py > out_'+str(j)
    print (s)
    os.system(s)
