import argparse
import json

import torch

import socket
import threading
import ctypes
import inspect

import datasets
from client import Client
from graph_struct import GraphStruct
from server import Server


class NodeServerRecv(threading.Thread):
    def __init__(self, _client_socket, _server):
        super(NodeServerRecv, self).__init__()
        self.client_socket = _client_socket
        self.server = _server

    def run(self):
        while True:
            _data = self.client_socket.recv(1024)
            # 客户端调用close终端tcp连接
            if _data == b'':
                self.stop_thread()

            data = json.loads(_data)
            print(data)

    def _async_raise(self, tid, exctype):
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self):
        self._async_raise(self.ident, SystemExit)


class NodeServerSocket:
    def __init__(self, _ip, _port, _server):
        self.ip = _ip
        self.port = _port
        self.server = _server
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

        while True:
            server_socket, client_addr = self.socket.accept()
            serverrecv = NodeServerRecv(server_socket, self.server)
            serverrecv.start()
            print("connect success...")


class NodeClientSocket:
    def __init__(self, _ip, _port, _client):
        self.socket = None
        self.ip = _ip
        self.port = _port
        self.client = _client
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))

    def send(self, msg):
        msg_bytes = bytes(msg, "utf-8")
        self.socket.send(msg_bytes)

    def close(self):
        self.socket.close()


def server_send(_ip, _port, _msg):
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _socket.connect((_ip, _port))
    if type(_msg) == str:
        _socket.send(_msg.encode('utf-8'))
    elif type(_msg) == list:
        _socket.send(json.dumps(_msg).encode('utf-8'))
    elif type(_msg) == dict:
        _socket.send(json.dumps(_msg).encode('utf-8'))
    else:
        pass
    _socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)
    train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"])
    server = Server(conf, eval_datasets)
    node_server = NodeServerSocket(args.ip, args.port, server)

    clients = []
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, str(c + 1)))
    generate_graph = GraphStruct()

    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = []
        for i in range(5):
            candidates.append(clients[i])
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if int(candidates[j].client_id) < int(candidates[i].client_id):
                    temp = candidates[i]
                    candidates[i] = candidates[j]
                    candidates[j] = temp
        candidates_dict = {}
        for c in candidates:
            candidates_dict[c.client_id] = c
        for c in candidates:
            c.client_dict = candidates_dict
            c.client_list = candidates
        server.client_dict = candidates_dict
        server.client_list = candidates

        generate_graph.node_number = len(candidates)
        communication_cost = c.compute_communication_cost()
        for c in candidates:
            generate_graph.communication_cost(communication_cost)
        generate_graph.init_graph(candidates)

        # 将生成树拓扑结构发送给客户端
        for i in range(5):
            server_send(conf["device" + str(i + 1)] + "_ip", conf["device" + str(i + 1) + "_port"],
                        "part connect graph")
            server_send(conf["device"+str(i+1)]+"_ip", conf["device"+str(i+1)+"_port"],
                        generate_graph.part_connect_graph)
        # 广播公钥
        for i in range(5):
            server_send(conf["device" + str(i + 1)] + "_ip", conf["device" + str(i + 1) + "_port"],
                        "advertise pubkey")
        # 共享密钥和bu
        for i in range(5):
            server_send(conf["device" + str(i + 1)] + "_ip", conf["device" + str(i + 1) + "_port"],
                        "shared key")
        # 开始联邦学习
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        for c in candidates:
            diff = c.local_train(server.global_model)
            c.mask(diff)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
        server.model_aggregate(weight_accumulator)
        for c in candidates:
            server.collect_shared_secretkey_bu({c.client_id: c.client_shared_key_bu})
        server.unmask()
        acc, loss = server.model_eval()
        print("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
