import random
n=random.randint(8,8)
t=n
n=2**n
print n
A=list()
B=list()
for i in range(n):
	A.append(random.randint(-1000,1000))
for i in range(n):
	B.append(random.randint(-1000,1000))

A.sort()
B.sort()

for x in A:
	print x, 
print  
for x in B:
	print x, 
print 
print 2**random.randint(1,t)