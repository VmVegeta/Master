import socket


def main():
    client_socket = socket.socket()
    host = '127.0.0.1'
    port = 1233

    print('Waiting for connection')
    try:
        client_socket.connect((host, port))
    except socket.error as e:
        print(str(e))

    response = client_socket.recv(1024)
    while True:
        something = input('Say Something: ')
        if something == '' or something is None:
            break
        client_socket.send(str.encode(something))
        response = client_socket.recv(1024)
        print(response.decode('utf-8'))

    client_socket.close()


if __name__ == '__main__':
    main()