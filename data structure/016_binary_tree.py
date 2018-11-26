# coding:utf-8

class Node(object):
    """"""
    def __init__(self, item):
        self.elem = item
        self.lchild = None
        self.rchild = None

class Tree(object):
    """二叉树"""
    def __init__(self):
        self.root = None

    def add(self, item):
        node = Node(item)
        if self.root is None:
            self.root = node
            return
        queue = [self.root]# 为方便理解,直接用列表实现队列的操作
        while queue:  # 只要队列不为空,就可以进行取出pop(0)处理判断添加
            cur_node = queue.pop(0)
            if cur_node.lchild is None:
                cur_node.lchild = node
                return    #退出
            else:
                queue.append(cur_node.lchild)# 如果左节点存在,则追加到队列中,以备下次处理.
            if cur_node.rchild is None:
                cur_node.rchild = node
                return   #添加到右节点后退出
            else:
                queue.append(cur_node.rchild)# 如果右节点存在,则追加到队列中,以备下次处理.

    def breadth_travel(self):
        """广度遍历"""
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            print(cur_node.elem, end=" ")
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

    def preorder(self, node):
        """先序遍历"""
        if node is None:
            return
        print(node.elem, end=" ")
        self.preorder(node.lchild)
        self.preorder(node.rchild)

    def inorder(self, node):
        """中序遍历"""
        if node is None:
            return
        self.inorder(node.lchild)
        print(node.elem, end=" ")
        self.inorder(node.rchild)

    def postorder(self, node):
        """后序遍历"""
        if node is None:
            return
        self.postorder(node.lchild)
        self.postorder(node.rchild)
        print(node.elem, end=" ")


if __name__ == "__main__":
    tree = Tree()
    tree.add(0)
    tree.add(1)
    tree.add(2)
    tree.add(3)
    tree.add(4)
    tree.add(5)
    tree.add(6)
    tree.add(7)
    tree.add(8)
    tree.add(9)
    tree.breadth_travel()
    print(" ")
    tree.preorder(tree.root)
    print(" ")
    tree.inorder(tree.root)
    print(" ")
    tree.postorder(tree.root)
    print(" ")