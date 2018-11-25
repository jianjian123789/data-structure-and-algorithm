# coding:utf-8


class DoubleQueue(object):
    """队列"""
    def __init__(self):
        self.__list=[]

    def add_front(self,item):
        """从队列的头部添加元素"""
        self.__list.insert(0,item)

    def add_rear(self,item):
        """入队"""
        self.__list.append(item)

    def pop_front(self):
        """从队列头部取出元素"""
        return  self.__list.pop(0)

    def pop_rear(self):
        """从队列尾部取出元素"""
        return  self.__list.pop()

    def is_empty(self):
        """判断是否为空"""
        return self.__list==[]

    def size(self):
        """返回队列大小"""
        return len(self.__list)

if __name__=='__main__':
    dq=DoubleQueue()
    dq.add_front(1)
    dq.add_front(2)
    dq.add_front(3)
    print(dq.pop_front())
    print (dq.pop_front())
    print (dq.pop_front())
