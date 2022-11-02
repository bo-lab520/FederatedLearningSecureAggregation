from socket import *

server_port = 8001

server_socket = socket(AF_INET, SOCK_STREAM)
server_socket.bind(('', server_port))
# 服务器最大连接2个请求
server_socket.listen(2)
while True:
    print("receive data:")
    data_socket, client_addr = server_socket.accept()  # 获取请求方的地址，并创建一个新的套接字data_socket用来接受数据。
    while True:
        data = data_socket.recv(4096).decode()
        print(data)
        data_socket.send(str("success").encode())
