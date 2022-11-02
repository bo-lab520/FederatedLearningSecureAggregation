from socket import *

def main():
    server_port = 1200
    server_socket = socket(AF_INET, SOCK_DGRAM)
    server_socket.bind(('', server_port))
    print('the server is ready to receive')
    while True:
        message, client_address = server_socket.recvfrom(2048)
        print(client_address)
        modifie_message = message.upper()
        server_socket.sendto(modifie_message, client_address)
        print('success!')

if __name__ == '__main__':
    main()