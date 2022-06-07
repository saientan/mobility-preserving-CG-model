f=open('all_files','r')
l_f=f.readlines()

l=len(l_f)
#
#file_name=[]
#maximum_r2=[]


file_name=[]
maximum_r2=[]
for i in range(0,l):
    print (i)
    s=''
    for j in range(0,len(l_f[i])-1):
        s=s+l_f[i][j]
    g=open(s,'r')
    l_g=g.readlines()
    lg=len(l_g)
    a=[]
    b=[]
    for j in range(0,lg):
        if "FULL_test" in l_g[j]:
            a.append(j)

    for j in range(0,len(a)):
        line=l_g[a[j]]
        b.append(float(line.split()[1]))
    file_name.append(s)
    maximum_r2.append(sum(b)/len(b))
    

for i in range(0,len(file_name)):
    if maximum_r2[i]==max(maximum_r2):
        print (file_name[i], maximum_r2[i])





