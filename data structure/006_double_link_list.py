# coding:utf-8

class Node(object):
    """节点"""
    def __init__(self):
        self.elem=item
        self.next=None
        self.prev=None


#链表类用来将节点Node进行连接串行
class DoubleLinkList(object):
    """单链表"""

    def __init__(self,node=None):#node=None,增加一个节点的默认属性,保证没有节点传入时头指针为空
        #有节点时我们的头指针指向这个节点
        # self.__head=None #这个属性是自己对象内部使用,对外不暴露,因此使用私有属性
        # #self.__head为None时,表示我们这个链表中没有任何一个节点
        self.__head=node #这个头指针指向我们初始化的节点


    def is_empty(self):
        """链表是否为空"""
        return self.__head==None

    def length(self):
        """链表长度"""
        #要注意空链表和非空链表的情况讨论验证
        cur = self.__head  # cur游标,用来移动遍历节点,初始指向头节点(头指针)
        count = 0  # count用来记录数量
        while cur != None:
            count += 1
            cur = cur.next  # 将指针移动到下一个节点
        return count


    def travel(self):
        """链表遍历"""
        cur=self.__head
        while cur!=None:
            print(cur.elem,end=' ')
            cur=cur.next

    def add(self,item):
        """链表头部添加元素,头插法"""
        node=Node(item)
        node.next=self.__head
        self.__head=node
        node.next.prev=node

    def append(self,item):
        """链表尾部添加元素,尾插法"""
        node=Node(item)#用户传入一个元素后,我们在内部实现构造节点,同时加到链表中去
        if self.is_empty():#如果链表为空
            self.__head=node
        else:
            cur=self.__head
            while cur.next!=None:
                cur=cur.next
            cur.next=node
            node.prev=cur

    def insert(self,pos,item):
        """在指定位置添加元素"""
        #注意区分先后顺序操作;pos是从0开始的
        cur=self.__head
        if pos<=0:
            self.add(item)
        elif pos>(self.length()-1):
            self.append(item)
        else:
            count=0
            while count<pos:
                count+=1
                cur=cur.next
            #当循环退出后,cur指向pos的位置
            node=Node(item)
            node.next=cur
            node.prev=cur.prev
            cur.prev.next=node
            cur.prev=node

    def remove(self,item):
        """删除节点"""
        #要先找到search该节点,然后再进行删除
        cur =self.__head
        while cur!=None:
            if cur.elem==item:
                # 先判断此节点是否是头节点,如果只有一个头节点而且要删除这个头节点,那么需要进行额外的操作
                # 头节点
                if cur==self.__head:
                    self.__head=cur.next
                    if cur.next:
                        #判断链表是否只有一个节点,只有一个节点时不会执行该句,没有prev
                        cur.next.prev=None  #在头节点时
                else:
                    cur.prev.next=cur.next
                    if cur.next:
                        #判断是否删除最后的一个节点,最后一个节点的next是None,没有prev的.
                        cur.next.prev=cur.prev
                break
            else:
                cur=cur.next


    def search(self,item):
        """查找节点是否存在"""
        #通过遍历的方式进行查找
        cur=self.__head
        while cur!=None:
            if cur.elem==item:
                return True
            else:
                cur=cur.next
        return False

if __name__=="__main__":
    dll=DoubleLinkList()
