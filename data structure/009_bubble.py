# coding:utf-8

def bubble_sort(alist):
    """冒泡排序"""
    n=len(alist)
    for j in range(0,n-1):
        # 班长需要走的轮数
        count=0 #进行优化的操作:对于顺序的列表可以提升时间复杂度到O(n)
        for i in range(0,n-1-j):
            #班长从头走到尾
            if alist[i]>alist[i+1]:
                alist[i],alist[i+1]=alist[i+1],alist[i]
        if 0==count:
            return

a=[2,3,1,9,32,22]
bubble_sort(a)
print(a)