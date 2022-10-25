import json
import socket
import threading
import time

import numpy as np
import torch

import datasets
from server import Server

global weight_accumulator
# 记录已经收到的客户端模型数量
global reach_clients
# 一轮聚合完成？
global is_agg_ok
# 联邦学习结束？
global agg_over
weight_accumulator = {}
reach_clients = 0
is_agg_ok = False
agg_over = False


class Agg(threading.Thread):
    def __init__(self, _server):
        super(Agg, self).__init__()
        self.current_epoch = 0
        self.server = _server
        self.n_clients = self.server.conf["k"]

    def agg(self):
        global weight_accumulator
        global agg_over
        if self.current_epoch < self.server.conf["global_epochs"]:
            self.server.model_aggregate(weight_accumulator)
            acc, loss = self.server.model_eval()
            print("Global Epoch {}, acc: {}, loss: {}\n".format(self.current_epoch, acc, loss))
            self.current_epoch += 1

        if self.current_epoch == self.server.conf["global_epochs"]:
            # self.server.global_model.save("model.pth")
            agg_over = True
            print("federated learning over ... ")

    def run(self):
        global is_agg_ok
        global reach_clients
        while (1):
            if reach_clients == self.n_clients:
                self.agg()
                # 下一轮训练
                reach_clients = 0
                # 聚合完成
                is_agg_ok = True


class ClientRecv(threading.Thread):
    def __init__(self, _client_socket, _server):
        super(ClientRecv, self).__init__()
        self.client_socket = _client_socket
        self.server = _server
        self.TYPE_MAP = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
        }

    def send(self, msg):
        msg_bytes = bytes(msg, "utf-8")
        self.client_socket.send(msg_bytes)

    def recv(self):
        _data = self.client_socket.recv(1024)
        data = _data.decode()
        return data

    def send_param_to_client(self):
        # for name, data in self.server.global_model.state_dict().items():
        #     self.send(str(name) + ':' + str(data))
        pass

    def run(self):
        global weight_accumulator
        global reach_clients
        global is_agg_ok
        global agg_over
        signal = -1
        name = ""
        type = ""
        shape = (0,)
        params = b''

        for _name, _params in self.server.global_model.state_dict().items():
            weight_accumulator[_name] = torch.zeros_like(_params)

        self.send("start train")

        while (1):
            # print("server receive data:")
            # data = self.recv()
            _data = self.client_socket.recv(1024)
            data = ""
            # print(data)
            if signal == -1:
                data = _data.decode()
            elif signal == 0:
                data = _data.decode()
                name = data
                signal = -1
            elif signal == 1:
                data = _data.decode()
                type = data
                signal = -1
            elif signal == 2:
                data = _data.decode()
                if data == "":
                    shape = (0,)
                else:
                    shape = tuple(map(int, data.split(', ')))
                signal = -1
            elif signal == 3:
                if _data == b'grad end':
                    data = _data.decode()
                    signal = -1
                else:
                    params += _data
            else:
                np_param = np.frombuffer(params, dtype=self.TYPE_MAP[type]).reshape(shape)
                t_param = torch.tensor(np_param)
                weight_accumulator[name].add_(t_param)
                data = _data.decode()
                signal = -1

            if data == "start":
                pass
            elif data == "end":
                print(data)
                reach_clients += 1
                print(reach_clients)
                while (1):
                    if is_agg_ok:
                        # self.send_param_to_client()
                        for _name, _params in self.server.global_model.state_dict().items():
                            weight_accumulator[_name] = torch.zeros_like(_params)
                        break
                    time.sleep(0.1)

                is_agg_ok = False
                if agg_over:  # 联邦学习结束
                    self.send("agg over")
                    break
                else:
                    # 一轮聚合结束，重新下发模型梯度
                    self.send_param_to_client()
                    self.send("start train")

            elif data == "name start":
                signal = 0
            elif data == "name end":
                signal = -1
            elif data == "type start":
                signal = 1
            elif data == "type end":
                signal = -1
            elif data == "dim start":
                signal = 2
            elif data == "dim end":
                signal = -1
            elif data == "grad start":
                signal = 3
            elif data == "grad end":
                signal = 4

            else:
                pass

    def close(self):
        self.client_socket.close()


class Accept(threading.Thread):
    def __init__(self, _socket, _server):
        super(Accept, self).__init__()
        self.socket = _socket
        self.server = _server

    def run(self):
        while (1):
            self.client_socket, self.client_addr = self.socket.accept()
            clientrecv = ClientRecv(self.client_socket, self.server)
            clientrecv.start()
            print("connect success...")


class ServerSocket:
    def __init__(self, _ip, _port, _server):
        self.ip = _ip
        self.port = _port
        self.server = _server
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

        self.accept_thread = Accept(self.socket, self.server)
        self.accept_thread.start()

        self.agg=Agg(self.server)
        self.agg.start()

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)
    _, eval_datasets = datasets.get_dataset("../data/", conf["type"])
    server = Server(conf, eval_datasets)

    server_socket = ServerSocket("127.0.0.1", 8888, server)
