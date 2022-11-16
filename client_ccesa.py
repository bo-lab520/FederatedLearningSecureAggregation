import json
import time

import socket
import threading
import ctypes
import inspect

with open("utils/conf.json", 'r') as f:
    conf = json.load(f)


def client_send(_ip, _port, _msg_signal, _msg_data):
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
    time.sleep(0.5)
    if type(_msg_data) == str:
        _socket.send(_msg_data.encode('utf-8'))
    elif type(_msg_data) == list:
        _socket.send(json.dumps(_msg_data).encode('utf-8'))
    elif type(_msg_data) == dict:
        _socket.send(json.dumps(_msg_data).encode('utf-8'))
    else:
        pass
    time.sleep(0.5)
    _socket.close()


def send_pubkey_to_server(pubkey):
    client_send(conf["edge" + "_ip"],
                conf["edge" + "_port"],
                "pubkey", pubkey)


def send_part_secretkey_bu_to_server(part_secretkey_bu):
    client_send(conf["edge" + "_ip"],
                conf["edge" + "_port"],
                "part_secretkey_bu", part_secretkey_bu)


def send_shared_secretkey_bu_to_server(client_id, client_shared_key_bu):
    client_send(conf["edge" + "_ip"],
                conf["edge" + "_port"],
                "unmask", {client_id: client_shared_key_bu})


class DeviceServerRecv(threading.Thread):
    def __init__(self, _client_socket, _client):
        super(DeviceServerRecv, self).__init__()
        self.client_socket = _client_socket
        self.client = _client
        self.signal = 0
        self.last_id = "0"

    def run(self):
        while True:
            _data = self.client_socket.recv(1024)
            if _data == b'':
                self.stop_thread()
            elif _data == b'part connect graph':
                self.signal = 1
            elif _data == b'advertise pubkey':
                self.signal = 2
                send_pubkey_to_server({self.client.client_id: self.client.sec_agg.pubkey})
            elif _data == b'transmit_pubkey':
                self.signal = 3
            elif _data == b'shared key':
                self.signal = 4
                part_secretkey_bu = self.client.shared_secretkey_bu()
                send_part_secretkey_bu_to_server(part_secretkey_bu)
            elif _data == b'transmit_part_secretkey_bu':
                self.signal = 5
            elif _data == b'unmask':
                self.signal = 6
                send_shared_secretkey_bu_to_server(self.client.client_id,
                                                   self.client.client_shared_key_bu)
            else:
                if self.signal == 1:
                    data = json.loads(_data)
                    self.client.part_connect_graph = data
                elif self.signal == 3:
                    data = json.loads(_data)
                    self.client.store_pubkey(data)
                elif self.signal == 5:
                    data = json.loads(_data)
                    self.client.store_shared_secretkey_bu(data)
                else:
                    pass

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


class DeviceServerSocket(threading.Thread):
    def __init__(self, _ip, _port):
        super(DeviceServerSocket, self).__init__()
        self.ip = _ip
        self.port = _port
        self.client = None
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

    def set_client(self, _client):
        self.client = _client

    def run(self):
        while True:
            client_socket, client_addr = self.socket.accept()
            clientrecv = DeviceServerRecv(client_socket, self.client)
            clientrecv.start()
            print("device{} connect success...".format(self.client.client_id))
