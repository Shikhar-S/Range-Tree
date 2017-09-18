#checks output of merge algorithm

with open('g.txt','r') as G:
	for line in G:
		line=line.split()
		line=[int(x) for x in line]
		check_list=sorted(line)
		for x,y in zip(line,check_list):
			if(x != y):
				print 'WRONG'
				break
		print 'CORRECT'