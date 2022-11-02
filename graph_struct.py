from collections import defaultdict
from heapq import heapify, heappop, heappush

global cur_node_num
cur_node_num = 0

class GraphStruct(object):
    def __init__(self):
        self.node_number = 0
        self.client_communication_cost = []
        self.nodes = ''
        self.all_connect_graph = []
        self.part_connect_graph = []

    def communication_cost(self, communication_cost):
        global cur_node_num
        cur_node_num += 1
        if cur_node_num == self.node_number:
            # 构造全连通图
            # cost_graph = {}
            # for client_cost in self.client_communication_cost:
            #     for id1, id2, cost in client_cost:
            #         cost_graph[id1+id2] = (id1, id2, cost)
            # for item in cost_graph:
            #     self.all_connect_graph.append(cost_graph[item])
            edges = [("1", "2", 10), ("1", "3", 9), ("1", "4", 4), ("1", "5", 5),
                     ("2", "3", 8), ("2", "4", 6), ("2", "5", 11),
                     ("3", "4", 15), ("3", "5", 12),
                     ("4", "5", 7)]
            self.all_connect_graph = edges
        else:
            self.client_communication_cost.append(communication_cost)

    def generate_part_graph(self):
        # 构造最小树
        element = defaultdict(list)
        for start, stop, weight in self.all_connect_graph:
            element[start].append((weight, start, stop))
            element[stop].append((weight, stop, start))
        all_nodes = set(self.nodes)
        used_nodes = set(self.nodes[0])
        usable_edges = element[self.nodes[0]][:]
        heapify(usable_edges)
        # 建立最小堆
        MST = []
        while usable_edges and (all_nodes - used_nodes):
            weight, start, stop = heappop(usable_edges)
            if stop not in used_nodes:
                used_nodes.add(stop)
                MST.append((start, stop, weight))
                for member in element[stop]:
                    if member[2] not in used_nodes:
                        heappush(usable_edges, member)

        return MST

    def init_graph(self, candidates):
        # for c in candidates:
        #     self.nodes += str(c.client_id)
        self.nodes = '12345'
        self.part_connect_graph = self.generate_part_graph()

if __name__ == '__main__':
    g = GraphStruct()
    g.init_graph([])
    print(g.part_connect_graph)
