# coding:utf-8



class Stack(object):
    """栈"""
# 我们用顺序表来实现栈的操作,python中列表list的实现就是已经用顺序表进行封装的操作
    def __init__(self):
        self.__list=[]#创建一个私有容器,避免外部访问到而能够控制内部方法

    def push(self,item):
        """添加一个新的元素item到栈顶"""
        self.__list.append(item)#如果使用链表,则从头部添加元素,时间复杂度是O(n),尾部需要遍历整个元素
        #如果使用顺序表,则从尾部添加元素,时间复杂度是O(1)

    def pop(self):
        """弹出栈顶元素"""
        return  self.__list.pop()

    def peek(self):
        """返回栈顶元素"""
        if self.__list:
            return self.__list[-1]
        else:
            return None

    def is_empty(self):
        """判断栈是否为空"""
        return self.__list==[]
        # return not self.__list #这句话也可以实现

    def size(self):
        """返回栈的元素个数"""


if __name__=="__main__":
    s=Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())
