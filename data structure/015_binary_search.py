# coding:utf-8
# [17, 20, 26, 31, 44, 54, 55, 77, 93]
# mid = n/2
#
# [17, 20, 26, 31]
# mid = n/2


def binary_search(alist, item):
    """二分查找,递归版本,传入新的列表"""
    n = len(alist)
    if n > 0:
        mid = n//2
        if alist[mid] == item:
            return True
        elif item < alist[mid]:
            return binary_search(alist[:mid], item)#调用自身,需要返回自身正确与否
        else:
            return binary_search(alist[mid+1:], item)
    return False #对于空列表或者递归调用结束的标志,返回false



def binary_search_2(alist, item):
    """二分查找， 非递归,在原有的列表上进行操作"""
    n = len(alist)
    first = 0
    last = n-1
    while first <= last:
        mid = (first + last)//2
        if alist[mid] == item:
            return True
        elif item < alist[mid]:
            last = mid - 1
        else:
            first = mid + 1
    return False






if __name__ == "__main__":
    li = [17, 20, 26, 31, 44, 54, 55, 77, 93]
    print(binary_search(li, 55))
    print(binary_search(li, 100))
    print(binary_search_2(li, 55))
    print(binary_search_2(li, 100))
    print (binary_search([],1))