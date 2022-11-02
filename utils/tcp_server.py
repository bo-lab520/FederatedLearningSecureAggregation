import ctypes
import inspect
import socket
import threading
import json


class ClientRecv(threading.Thread):
    def __init__(self, _client_socket):
        super(ClientRecv, self).__init__()
        self.client_socket = _client_socket

    def run(self):
        while True:
            _data = self.client_socket.recv(1024)
            # 客户端调用close终端tcp连接
            if _data == b'':
                self.stop_thread()

            # data = json.loads(_data)
            data = _data.decode('utf-8')
            print(type(data))
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
    def __init__(self, _ip, _port):
        super(DeviceServerSocket, self).__init__()
        self.ip = _ip
        self.port = _port
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

        while True:
            client_socket, client_addr = self.socket.accept()
            clientrecv = ClientRecv(client_socket)
            clientrecv.start()
            print("connect success...")

if __name__ == '__main__':
    server = DeviceServerSocket("127.0.0.1", 8000)