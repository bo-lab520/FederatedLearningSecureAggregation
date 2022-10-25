import argparse
import json
import socket
import threading

import datasets
import models
from client import Client


class ClientSocket(threading.Thread):
    def __init__(self, _ip, _port, _client):
        super(ClientSocket, self).__init__()
        self.ip = _ip
        self.port = _port
        self.client = _client
        self.init()


    def init(self):
        # 创建套接字对象
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        print("client init success...")

    def send(self, msg):
        msg_bytes = bytes(msg, "utf-8")
        self.socket.send(msg_bytes)

    def recv(self):
        _data = self.socket.recv(1024)
        data = _data.decode()
        return data

    def run(self):
        while (1):
            print("client receive data:")
            data = self.recv()
            print(data)

            if data == "start train":
                diff = self.client.local_train(self.client.local_model)
                self.send("start")
                for name in diff:
                    np_params = diff[name].detach().numpy()
                    np_type = str(np_params.dtype)
                    np_dim = str(np_params.shape)[1:len(str(np_params.shape)) - 1]
                    self.send("name start")
                    self.send(name)
                    self.send("name end")
                    self.send("type start")
                    self.send(np_type)
                    self.send("type end")
                    self.send("dim start")
                    self.send(np_dim)
                    self.send("dim end")
                    self.send("grad start")
                    self.socket.send(np_params.tobytes())
                    self.send("grad end")
                self.send("end")
            elif data == "agg over":
                print("federated learning over ... ")
                break
            else:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=32)
    args = parser.parse_args()

    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)
    train_datasets, _ = datasets.get_dataset("../data/", conf["type"])
    client0 = Client(conf, models.get_model(conf["model_name"]), train_datasets, args.id)

    client_socket = ClientSocket("127.0.0.1", 8888, client0)
    client_socket.start()