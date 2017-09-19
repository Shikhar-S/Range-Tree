with open('g.txt','r') as G:
    for line in G:
        line=line.split()
        line=[int(x) for x in line]
        check_list=sorted(line)
        with open('f.txt','r') as F:
            n=0
            num_ele=0
            num_list=[]
            num_list_=[]
            for L in F:
                if(n==0):
                    num_ele=int(L)*2
                elif(n==1):
                    num_list=[int(x) for x in L.split()]
                elif(n==2):
                    num_list_=[int(x) for x in L.split()]
                    for x in num_list_:
                        num_list.append(x)
                    num_list.sort()
                n+=1
            for x,y in zip(check_list,num_list):
                if(x!=y):
                    print('Wrong')
print('correct')
