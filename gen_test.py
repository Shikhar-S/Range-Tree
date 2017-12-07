# -*- coding: utf8 -*-
from __future__ import print_function
import random
f=open('f.txt','w+')
for j in range(20):
	n=j
	n=2**int(n)
	lim=1000
	print(n,file=f)
	for i in range(n):
		x=random.randint(-lim,lim)
		y=random.randint(-lim,lim)
		print(x,end=' ',file=f)
		print(y,end='\n',file=f)
	print('Done with log-> ',j)
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
