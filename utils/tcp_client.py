from socket import *
import json

# msg = [(1,2,3),('1',2,3)]
msg = 1
socket_client = socket(AF_INET, SOCK_STREAM)
socket_client.connect(("127.0.0.1", 8000))
while True:
    data = input("please input:")
    _data = json.dumps(msg).encode('utf-8')
    print(_data)
    socket_client.send(_data)
    break
socket_client.close()

# socket_client = socket(AF_INET, SOCK_STREAM)
# socket_client.connect(("127.0.0.1", 8000))  # 建立TCP连接
# while True:
#     data = input("please input:")
#     socket_client.send(data.encode())  # 发送数据
#     break
# socket_client.close()
