import argparse
import json
import time

import torch

import socket
import threading
import ctypes
import inspect

import datasets
from client import Client
from client_eflsas import DeviceServerSocket
from graph import GraphStruct
from server import Server

global collect_nums
collect_nums = 0


def finish_step1(candidates):
    while True:
        is_finish = False
        for i in range(len(candidates)):
            if len(candidates[i].part_connect_graph) == 0:
                break
            if i == len(candidates) - 1:
                is_finish = True
        if is_finish:
            break
        time.sleep(1)


def finish_step2(candidates):
    while True:
        is_finish = False
        for i in range(len(candidates)):
            if len(candidates[i].client_pubkey) < len(candidates):
                break
            if i == len(candidates) - 1:
                is_finish = True
        if is_finish:
            break
        time.sleep(1)


def finish_step3(candidates):
    while True:
        is_finish = False
        for i in range(len(candidates)):
            if len(candidates[i].client_shared_key_bu) < len(candidates):
                break
            if i == len(candidates) - 1:
                is_finish = True
        if is_finish:
            break
        time.sleep(1)


def finish_step4():
    global collect_nums
    while True:
        if collect_nums == 5:
            collect_nums = 0
            break
        time.sleep(1)


def server_send(_ip, _port, _msg_signal, _msg_data):
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _socket.connect((_ip, _port))
    if type(_msg_signal) == str:
        _socket.send(_msg_signal.encode('utf-8'))
    elif type(_msg_signal) == list:
        _socket.send(json.dumps(_msg_signal).encode('utf-8'))
    elif type(_msg_signal) == dict:
        _socket.send(json.dumps(_msg_signal).encode('utf-8'))
    else:
        pass
    time.sleep(1)
    if type(_msg_data) == str:
        _socket.send(_msg_data.encode('utf-8'))
    elif type(_msg_data) == list:
        _socket.send(json.dumps(_msg_data).encode('utf-8'))
    elif type(_msg_data) == dict:
        _socket.send(json.dumps(_msg_data).encode('utf-8'))
    else:
        pass
    time.sleep(1)
    _socket.close()


class NodeServerRecv(threading.Thread):
    def __init__(self, _client_socket, _server):
        super(NodeServerRecv, self).__init__()
        self.client_socket = _client_socket
        self.server = _server
        self.signal = 0

    def run(self):
        global collect_nums
        while True:
            _data = self.client_socket.recv(1024)
            # 客户端调用close终端tcp连接
            if _data == b'':
                self.stop_thread()
            elif _data == b'unmask':
                self.signal = 1
            else:
                if self.signal == 1:
                    data = json.loads(_data)
                    self.server.collect_shared_secretkey_bu(data)
                    collect_nums += 1

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


class NodeServerSocket(threading.Thread):
    def __init__(self, _ip, _port, _server):
        super(NodeServerSocket, self).__init__()
        self.ip = _ip
        self.port = _port
        self.server = _server
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

    def run(self):
        while True:
            server_socket, client_addr = self.socket.accept()
            serverrecv = NodeServerRecv(server_socket, self.server)
            serverrecv.start()
            print("edge connect success...")


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    with open("utils/conf.json", 'r') as f:
        conf = json.load(f)
    train_datasets, eval_datasets = datasets.get_dataset("data/", conf["type"])

    # 启动服务器
    server = Server(conf, eval_datasets)
    node_server = NodeServerSocket(args.ip, args.port, server)
    node_server.start()

    device_servers = []
    for i in range(conf["k"]):
        device_server = DeviceServerSocket(conf["device" + str(i + 1) + "_ip"],
                                           conf["device" + str(i + 1) + "_port"])
        device_server.start()
        device_servers.append(device_server)

    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = []
        for i in range(conf["k"]):
            c = Client(conf, server.global_model, train_datasets, str(i + 1))
            candidates.append(c)
            device_servers[i].set_client(c)
        candidates_dict = {}
        for c in candidates:
            candidates_dict[c.client_id] = c
        for c in candidates:
            c.client_dict = candidates_dict
            c.client_list = candidates
        server.client_dict = candidates_dict
        server.client_list = candidates

        # 客户端传输通信时延 服务器计算拓扑结构
        # 计算...
        generate_graph = GraphStruct(3)
        generate_graph.communication_cost([])
        generate_graph.init_graph(candidates)

        # 将生成树拓扑结构发送给客户端
        print("下发拓扑图结构...")
        for c in candidates:
            server_send(conf["device" + c.client_id + "_ip"],
                        conf["device" + c.client_id + "_port"],
                        "part connect graph",
                        generate_graph.part_connect_graph)
        finish_step1(candidates)
        print("完成下发拓扑图结构...")
        # 广播公钥
        print("开始广播公钥...")
        for c in candidates:
            server_send(conf["device" + c.client_id + "_ip"],
                        conf["device" + c.client_id + "_port"],
                        "advertise pubkey", [])
        finish_step2(candidates)
        print("完成广播公钥...")

        # 共享密钥和bu
        print("开始共享密钥和bu...")
        for c in candidates:
            server_send(conf["device" + c.client_id + "_ip"],
                        conf["device" + c.client_id + "_port"],
                        "shared key", [])
        finish_step3(candidates)
        print("完成共享密钥和bu...")

        # for c in candidates:
        #     server_send(conf["device" + c.client_id + "_ip"],
        #                 conf["device" + c.client_id + "_port"],
        #                 "unmask", [])
        # print("------原始数据------")
        # for c in candidates:
        #     print(c.client_id, c.sec_agg.secretkey, c.sec_agg.sndkey)
        # print("------重构数据------")
        # for client_id in server.all_part_secretkey_bu:
        #     secretkey_bu = server.reconstruct_secretkey_bu(server.conf["t"], server.all_part_secretkey_bu[client_id])
        #     print(client_id, secretkey_bu[0], secretkey_bu[1])

        print("开始联邦学习...")
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        for c in candidates:
            diff = c.local_train(server.global_model)
            c.mask(diff)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)

        # 消除掩码
        for c in candidates:
            server_send(conf["device" + c.client_id + "_ip"],
                        conf["device" + c.client_id + "_port"],
                        "unmask", [])
        finish_step4()

        server.unmask()

        acc, loss = server.model_eval()
        print("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

        # clean server
        server.part_connect_graph = []
        server.all_part_secretkey_bu = {}
        server.client_dict = {}
        server.client_list = []

    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
