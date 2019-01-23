# coding:utf-8

with open('poems.txt','r') as f:
	line=f.readlines()
	for i in line[:5]:

		print(i)
#		print('--------------------')

		title,content=i.split(":")
		print(title,'========',content)
