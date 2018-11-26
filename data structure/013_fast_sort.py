# coding:utf-8

#first 和last表示操作列表的起始和终止部分
def fast_sort(alist,first,last):
    """快速排序"""
    #只有一个元素时,结束递归的操作
    if first>=last:
        return

    n=len(alist)
    mid_value=alist[first]
    low=first
    high=last
    while low<high:
        # high 左移
        while low<high and alist[high]>=mid_value:
            high-=1
        alist[low]=alist[high]

        # low 右移
        while low<high and alist[low]<mid_value:
            low+=1
        alist[high]=alist[low]
    # 从循环退出时,low==high
    alist[low]=mid_value

    # 对low左边的列表执行快速排序
    fast_sort(alist,first,low-1)#注意:我们必须传入以前的列表,不能传入新的列表切片,否则失去了与原列表关系
    # 对low右边的列表执行快速排序
    fast_sort(alist,low+1,last)

alist=[13,1,12,33,122,53]
print (alist)
fast_sort(alist,0,len(alist)-1)
print (alist)