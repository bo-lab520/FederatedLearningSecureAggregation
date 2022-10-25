import argparse
import json
import threading

import zmq

import datasets
import models
from client import Client


class ClientZMQ(threading.Thread):
    def __init__(self, _ip, _port, _client):
        super(ClientZMQ, self).__init__()
        self.ip = _ip
        self.port = _port
        self.client = _client
        self.init()

    def init(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        # self.socket.connect('tcp://127.0.0.1:8888')
        self.socket.connect("tcp://" + str(self.ip) + ":" + str(self.port))
        print("client init success...")

    def send(self, msg):
        msg_bytes = bytes(msg, "utf-8")
        self.socket.send(msg_bytes)

    def recv(self):
        _data = self.socket.recv()
        data = _data.decode()
        return data

    def run(self):
        self.send("connect")
        while True:
            # print("client receive data:")
            data = self.socket.recv()

            if data == b'start train':
                diff = self.client.local_train(self.client.local_model)
                self.send("start")
                _ = self.socket.recv()
                for name in diff:
                    # print(diff[name])
                    np_params = diff[name].detach().numpy()
                    np_name = str(name)
                    np_type = str(np_params.dtype)
                    np_dim = str(np_params.shape)[1:len(str(np_params.shape)) - 1]

                    b_np_params = np_params.tobytes()
                    b_np_name = bytes(np_name, "utf-8")
                    b_np_type = bytes(np_type, "utf-8")
                    b_np_dim = bytes(np_dim, "utf-8")
                    msg = b_np_name+b'+'+b_np_type+b'+'+b_np_dim+b'+'+b_np_params
                    self.socket.send(msg)
                    _ = self.socket.recv()

                self.send("end")

            elif data == b'agg over':
                print("federated learning over ... ")
                break
            elif data == b'':
                pass

        self.close()

    def close(self):
        self.socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=32)
    args = parser.parse_args()

    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)
    train_datasets, _ = datasets.get_dataset("../data/", conf["type"])
    client0 = Client(conf, models.get_model(conf["model_name"]), train_datasets, 2)
    client1 = Client(conf, models.get_model(conf["model_name"]), train_datasets, 6)

    client_zmq0 = ClientZMQ("127.0.0.1", 8080, client0)
    client_zmq0.start()
    client_zmq1 = ClientZMQ("127.0.0.1", 8081, client1)
    client_zmq1.start()

