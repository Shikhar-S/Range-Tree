import random
n=input()
n=2**n
lim=1000
print n
for i in range(n):
	x=random.randint(-lim,lim)
	y=random.randint(-lim,lim)
	print x, y
'''
queries=random.randint(1,10)
print queries
for _ in range(queries):
	x_lo=random.randint(-lim,lim)
	x_up=random.randint(-lim,lim)
	y_lo=random.randint(-lim,lim)
	y_up=random.randint(-lim,lim)
	if(x_lo > x_up ):
		x_lo, x_up = x_up,x_lo
	if(y_lo > y_up):
		y_lo, y_up = y_up, y_lo
	print x_lo, y_lo
	print x_up, y_up

'''