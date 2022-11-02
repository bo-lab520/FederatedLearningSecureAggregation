import argparse
import json
import socket
import threading
import ctypes
import inspect

import datasets
import models
from client import Client

# 1代表下一个数据为服务器发过来的最小生成树拓扑图
# 3代表下一个数据为密钥分发

with open("../utils/conf.json", 'r') as f:
    conf = json.load(f)


def client_send(_ip, _port, _msg):
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _socket.connect((_ip, _port))
    if type(_msg) == str:
        _socket.send(_msg.encode('utf-8'))
    elif type(_msg) == int:
        _socket.send(_msg.encode('utf-8'))
    elif type(_msg) == list:
        _socket.send(json.dumps(_msg).encode('utf-8'))
    elif type(_msg) == dict:
        _socket.send(json.dumps(_msg).encode('utf-8'))
    else:
        pass
    _socket.close()


def send_part_secretkey_bu_to_adj(client_id, part_secretkey_bu, part_connect_graph):
    for client1, client2, cost in part_connect_graph:
        if client1 == client_id:
            client_send(conf["device" + client2 + "_ip"],
                        conf["device" + client2 + "_port"],
                        client1 + "_part_secretkey_bu")
            client_send(conf["device" + client2 + "_ip"],
                        conf["device" + client2 + "_port"],
                        part_secretkey_bu)
        if client2 == client_id:
            client_send(conf["device" + client1 + "_ip"],
                        conf["device" + client1 + "_port"],
                        client2 + "_part_secretkey_bu")
            client_send(conf["device" + client1 + "_ip"],
                        conf["device" + client1 + "_port"],
                        part_secretkey_bu)


def transmit_part_secretkey_bu_to_adj(part_msg, last_id, client_id, part_connect_graph):
    for client1, client2, cost in part_connect_graph:
        if client1 == client_id:
            if client2 == last_id:
                pass
            else:
                client_send(conf["device" + client2 + "_ip"],
                            conf["device" + client2 + "_port"],
                            client1 + "_part_secretkey_bu")
                client_send(conf["device" + client2 + "_ip"],
                            conf["device" + client2 + "_port"],
                            part_msg)
        if client2 == client_id:
            if client1 == last_id:
                pass
            else:
                client_send(conf["device" + client1 + "_ip"],
                            conf["device" + client1 + "_port"],
                            client2 + "_part_secretkey_bu")
                client_send(conf["device" + client1 + "_ip"],
                            conf["device" + client1 + "_port"],
                            part_msg)


def transmit_pubkey_to_adj(pubkey, last_id, client_id, part_connect_graph):
    for client1, client2, cost in part_connect_graph:
        if client1 == client_id:
            if client2 == last_id:
                pass
            else:
                client_send(conf["device" + client2 + "_ip"],
                            conf["device" + client2 + "_port"],
                            client1 + "_pubkey")
                client_send(conf["device" + client2 + "_ip"],
                            conf["device" + client2 + "_port"],
                            pubkey)
        if client2 == client_id:
            if client1 == last_id:
                pass
            else:
                client_send(conf["device" + client1 + "_ip"],
                            conf["device" + client1 + "_port"],
                            client2 + "_pubkey")
                client_send(conf["device" + client1 + "_ip"],
                            conf["device" + client1 + "_port"],
                            pubkey)


def send_pubkey_to_adj(client_id, pubkey, part_connect_graph):
    for client1, client2, cost in part_connect_graph:
        if client1 == client_id:
            client_send(conf["device" + client2 + "_ip"],
                        conf["device" + client2 + "_port"],
                        client1 + "_pubkey")
            client_send(conf["device" + client2 + "_ip"],
                        conf["device" + client2 + "_port"],
                        pubkey)
        if client2 == client_id:
            client_send(conf["device" + client1 + "_ip"],
                        conf["device" + client1 + "_port"],
                        client2 + "_pubkey")
            client_send(conf["device" + client1 + "_ip"],
                        conf["device" + client1 + "_port"],
                        pubkey)


class DeviceServerRecv(threading.Thread):
    def __init__(self, _client_socket, _client):
        super(DeviceServerRecv, self).__init__()
        self.client_socket = _client_socket
        self.client = _client
        self.signal = 0
        self.last_id = "0"

    def run(self):
        global signal
        while True:
            _data = self.client_socket.recv(1024)
            # 客户端调用close终端tcp连接
            if _data == b'':
                self.stop_thread()
            elif _data == b'part connect graph':
                self.signal = 1
            elif _data == b'advertise pubkey':
                self.signal = 2
                send_pubkey_to_adj(self.client.client_id, {self.client.client_id: self.client.sec_agg.pubkey},
                                   self.client.part_connect_graph)
            elif _data[1:] == b'_pubkey':
                self.signal = 3
                self.last_id = _data[0:1].decode('utf-8')
            elif _data == b'shared key':
                self.signal = 4
                part_secretkey_bu = self.client.shared_secretkey_bu()
                send_part_secretkey_bu_to_adj(self.client.client_id, part_secretkey_bu,
                                              self.client.part_connect_graph)
            elif _data[1:] == b'_part secretkey bu':
                self.signal = 5
                self.last_id = _data[0:1].decode('utf-8')
            else:
                if self.signal == 1:
                    data = json.loads(_data)
                    self.client.part_connect_graph = data
                elif self.signal == 3:
                    data = json.loads(_data)
                    self.client.store_pubkey(data)
                    transmit_pubkey_to_adj(data, self.last_id, self.client.client_id,
                                           self.client.part_connect_graph)
                elif signal == 5:
                    data = json.loads(_data)
                    self.client.store_shared_secretkey_bu(data)
                    transmit_part_secretkey_bu_to_adj(data, self.last_id, self.client.client_id,
                                                      self.client.part_connect_graph)

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


class DeviceServerSocket:
    def __init__(self, _ip, _port, _client):
        super(DeviceServerSocket, self).__init__()
        self.ip = _ip
        self.port = _port
        self.client = _client
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

        while True:
            client_socket, client_addr = self.socket.accept()
            clientrecv = DeviceServerRecv(client_socket, self.client)
            clientrecv.start()
            print("connect success...")


class DeviceClientSocket:
    def __init__(self, _ip, _port):
        self.socket = None
        self.ip = _ip
        self.port = _port
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))

    def send(self, msg):
        msg_bytes = bytes(msg, "utf-8")
        self.socket.send(msg_bytes)

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--id', type=int, default=-1)
    args = parser.parse_args()

    train_datasets, _ = datasets.get_dataset("../data/", conf["type"])
    client = Client(conf, models.get_model(conf["model_name"]), train_datasets, args.id)

    device_server = DeviceServerSocket(args.ip, args.port, client)
