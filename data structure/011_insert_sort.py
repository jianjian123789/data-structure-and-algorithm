# coding:utf-8

def insert_sort(alist):
    """插入排序"""
    n=len(alist)
    # 从右边无序列表中取出多少个元素执行这样的过程
    for j in range(1,n):
        # i 代表的是内层循环的起始值
        i=j
        # 执行从右边的无序列表中取出第一个元素,即i位置的元素,然后将其插入到前面正确的位置中
        while i>0:
            if alist[i]<alist[i-1]:
                alist[i],alist[i-1]=alist[i-1],alist[i]
                i-=1
            else:
                break
alist=[1,123,12,22,44,125,32,33]
print(alist)
insert_sort(alist)
print(alist)