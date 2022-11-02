import time
from socket import *
#127.0.0.1是本地回环地址，经常用来进行测试
def main():
    server_address = '127.0.0.1'
    server_port = 1200
    client_socket = socket(AF_INET, SOCK_DGRAM)
    while True:
        message = input('Input lowecase senctece: ')
        client_socket.sendto("123".encode(), (server_address, server_port))
        modified_message, server_address = client_socket.recvfrom(2048)
        print(modified_message)
    client_socket.close()
if __name__ == '__main__':
    main()