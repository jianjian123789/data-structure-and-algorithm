# coding:utf-8

def select_sort(alist):
    """选择排序"""
    n=len(alist)
    for j in range(n-1):# j:0-n-2
        min_index=j
        for i in range(j+1,n):
            #从无序列表中取出最小值操作
            if alist[min_index]>alist[i]:
                min_index=i
        alist[j],alist[min_index]=alist[min_index],alist[j]


alist=[123,12,33,121,44,22,1,4]
print (alist)
select_sort(alist)
print (alist)
