import json
import threading
import time

import numpy as np
import torch
import zmq

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
        # print(weight_accumulator)
        if self.current_epoch < self.server.conf["global_epochs"]:
            self.server.model_aggregate(weight_accumulator)
            # for name, params in self.server.global_model.state_dict().items():
            #     print(name)
            #     print(params)
            acc, loss = self.server.model_eval()
            print("Global Epoch {}, acc: {}, loss: {}\n".format(self.current_epoch, acc, loss))
            self.current_epoch += 1

        if self.current_epoch == self.server.conf["global_epochs"]:
            torch.save(self.server.global_model, "../model.pth")
            agg_over = True
            print("federated learning over ... ")

    def run(self):
        global is_agg_ok
        global reach_clients
        while True:
            if reach_clients == self.n_clients:
                self.agg()
                # 下一轮训练
                reach_clients = 0
                # 聚合完成
                is_agg_ok = True
            time.sleep(1)


class ClientZMQ(threading.Thread):

    def __init__(self, _ip, _port, _server):
        super(ClientZMQ, self).__init__()
        self.socket = None
        self.ip = _ip
        self.port = _port
        self.server = _server
        self.TYPE_MAP = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
        }
        self.init()

    def init(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        # socket.bind('tcp://127.0.0.1:8888')
        self.socket.bind("tcp://" + str(self.ip) + ":" + str(self.port))

    def send(self, msg):
        msg_bytes = bytes(msg, "utf-8")
        self.socket.send(msg_bytes)

    def recv(self):
        _data = self.socket.recv(1024)
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

        for _name, _params in self.server.global_model.state_dict().items():
            weight_accumulator[_name] = torch.zeros_like(_params)

        while True:
            # print("server receive data:")
            _data = self.socket.recv()
            if _data == b'connect':
                self.send("start train")
            elif _data == b'start':
                self.send("")
            elif _data == b'end':
                reach_clients += 1
                print(reach_clients)
                while True:
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
            else:
                bytes_msg = _data.split(b'+')
                name = bytes_msg[0].decode()
                _type = bytes_msg[1].decode()
                dim = bytes_msg[2].decode()
                if dim == "":
                    shape = ()
                elif dim[len(dim)-1] == ',':
                    _dim = int(dim[0:len(dim)-1])
                    shape = (_dim,)
                else:
                    shape = tuple(map(int, dim.split(', ')))
                np_param = np.frombuffer(bytes_msg[3], dtype=self.TYPE_MAP[_type]).reshape(shape)
                t_param = torch.tensor(np_param)
                # print(t_param)
                weight_accumulator[name].add_(t_param)
                self.send("")

        self.close()

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)
    _, eval_datasets = datasets.get_dataset("../data/", conf["type"])
    server = Server(conf, eval_datasets)

    server_zmq0 = ClientZMQ("127.0.0.1", 8080, server)
    server_zmq0.start()
    server_zmq1 = ClientZMQ("127.0.0.1", 8081, server)
    server_zmq1.start()
    agg = Agg(server)
    agg.start()