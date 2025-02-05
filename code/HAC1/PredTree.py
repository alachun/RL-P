from anytree import Node, RenderTree
from anytree.exporter import DotExporter

class PredTree:
    def __init__(self):
        self.root = Node("root")
        self.path_to_index = {}
        self.index_to_path = {}

    def create_obs_tree(self, pred_list, objclass_dict, obj_dict):
        # construct obs tree
        for pred in pred_list:
            pred_node = Node(pred, parent=self.root)
            # 用于存储上一层的节点，以便将当前层的节点挂载到它们下
            previous_layer_nodes = [pred_node]

            # 遍历每一层
            for layer in objclass_dict[pred]:
                current_layer_nodes = []
                for parent in previous_layer_nodes:
                    for node_name in obj_dict.get(layer, []):
                        # 创建当前层的节点，并将其设置为 parent 的子节点
                        child = Node(f"{node_name}", parent=parent)
                        current_layer_nodes.append(child)
                # 更新上一层的节点为当前层的节点
                previous_layer_nodes = current_layer_nodes

        # 打印树结构（可选）
        '''for pre, fill, node in RenderTree(self.root):
            print(f"{pre}{node.name}")'''

        current_index = 0
        for leaf in self.root.leaves:
            # 获取从根到叶子的路径
            path = [ancestor.name for ancestor in leaf.path]
            path_tuple = tuple(path)  # 可以使用元组作为键
            self.path_to_index[path_tuple] = current_index
            self.index_to_path[current_index] = path_tuple
            current_index += 1

    def create_action_tree(self, literals):
        for i, literal in enumerate(literals):
            self.add_tree_path(i, self.get_tree_path(literal))

        # 打印树结构（可选）
        '''for pre, fill, node in RenderTree(self.root):
            print(f"{pre}{node.name}")'''


    def get_index_by_path(self, path_list):
        path_tuple = tuple(path_list)
        return self.path_to_index.get(path_tuple, None)

    # 示例：通过索引查找路径
    def get_path_by_index(self, index):
        return self.index_to_path.get(index, None)

    def get_tree_path(self, literal):
        path = ['root']
        path.append(str(literal.predicate))
        for var in literal.variables:
            obj, category = str(var).split(":")
            path.append(obj)
        return path

    # 定义添加路径的函数
    def add_tree_path(self, i, path):
        path = path[1:]
        current_node = self.root
        for name in path:
            # 查找当前节点的子节点中是否已有该名称的节点
            existing_children = [child for child in current_node.children if child.name == name]
            if existing_children:
                # 如果存在，移动到该子节点
                current_node = existing_children[0]
            else:
                # 如果不存在，创建新节点并移动到新节点
                new_node = Node(name, parent=current_node)
                new_node.index = i
                current_node = new_node
