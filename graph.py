import random
import time
from collections import defaultdict
from heapq import heapify, heappop, heappush

global cur_node_num
cur_node_num = 0


class GraphStruct(object):
    def __init__(self, _class):
        self.node_number = 0
        self.client_communication_cost = []
        self.nodes = ''
        self.all_connect_graph = []
        self.part_connect_graph = []
        self._class = _class

    def communication_cost(self, communication_cost):
        # global cur_node_num
        # cur_node_num += 1
        # if cur_node_num == self.node_number:
        #     # 构造全连通图
        #     # cost_graph = {}
        #     # for client_cost in self.client_communication_cost:
        #     #     for id1, id2, cost in client_cost:
        #     #         cost_graph[id1+id2] = (id1, id2, cost)
        #     # for item in cost_graph:
        #     #     self.all_connect_graph.append(cost_graph[item])
        #     edges = [("1", "2", 10), ("1", "3", 9), ("1", "4", 4), ("1", "5", 5),
        #              ("2", "3", 8), ("2", "4", 6), ("2", "5", 11),
        #              ("3", "4", 15), ("3", "5", 12),
        #              ("4", "5", 7)]
        #     self.all_connect_graph = edges
        # else:
        #     self.client_communication_cost.append(communication_cost)

        # test
        edges = [("1", "2", 10), ("1", "3", 9), ("1", "4", 4), ("1", "5", 5),
                 ("2", "3", 8), ("2", "4", 6), ("2", "5", 11),
                 ("3", "4", 15), ("3", "5", 12),
                 ("4", "5", 7)]
        self.all_connect_graph = edges

    def generate_mst_graph(self):
        element = defaultdict(list)
        for start, stop, weight in self.all_connect_graph:
            element[start].append((weight, start, stop))
            element[stop].append((weight, stop, start))
        all_nodes = set(self.nodes)
        used_nodes = set(self.nodes[0])
        usable_edges = element[self.nodes[0]][:]
        heapify(usable_edges)
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

    def generate_random_graph(self):
        random_graph = []
        while True:
            node_t = {}
            for _id in self.nodes:
                node_t[_id] = 0
            for e in self.all_connect_graph:
                if random.random() < 0.7:
                    random_graph.append(e)
                    node_t[e[0]] += 1
                    node_t[e[1]] += 1
                else:
                    continue
            is_success = False
            for _id in self.nodes:
                if node_t[_id] < 3:
                    break
                if _id == '5':
                    is_success = True
            if is_success:
                break
            time.sleep(1)
            random_graph = []
        return random_graph

    def init_graph(self, candidates):
        # for c in candidates:
        #     self.nodes += str(c.client_id)
        self.nodes = '12345'
        if self._class == 1:
            # 全连通图
            self.part_connect_graph = self.all_connect_graph
        elif self._class == 2:
            # 随机图
            self.part_connect_graph = self.generate_random_graph()
        else:
            # 最小生成树
            self.part_connect_graph = self.generate_mst_graph()


if __name__ == '__main__':
    g = GraphStruct()
    g.communication_cost([])
    g.init_graph([])
    print(g.part_connect_graph)
