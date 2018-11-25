# coding:utf-8

def shell_sort(alist):
    """希尔排序"""

    n=len(alist)
    gap=n//2
    # gap变化到0之前,插入sauna执行的次数
    while gap>=1:
        # 插入算法,与普通插入算法的区别就是gap的步长
        for j in range(gap,n):
            i=j
            while i>0:
                if alist[i]<alist[i-gap]:
                    alist[i],alist[i-gap]=alist[i-gap],alist[i]
                    i-=gap
                else:
                    break
        # 缩短gap的长度
        gap//=2


alist=[1,12,42,122,32]
print(alist)
shell_sort(alist)
print (alist)
