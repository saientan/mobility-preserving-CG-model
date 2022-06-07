import os
for i in range(1,23):
    j=i*10
    s1='shuffled_text_removed_all_data.dat'
    s="head -"+str(j)+" "+s1+" > learning_data.dat"
    print (s)
    os.system(s)
    s='python NN_with_optimized_parameter_KFOLD.py > out_'+str(j)
    print (s)
    os.system(s)
