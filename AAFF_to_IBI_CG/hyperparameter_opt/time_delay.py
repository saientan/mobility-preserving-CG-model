import time
import os


N=10000

for i in range(0,N):
    time.sleep(10)
    s="squeue -u kn1269 > tot_job"
    os.system(s)
    f=open('tot_job','r')
    l_f=f.readlines()
    l=len(l_f)
    if l<20:
        s="sbatch sub.sh"
        os.system(s)
       

